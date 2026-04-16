"""
DINA VB Q-Matrix Estimator
Translation of Oka & Okada (2023) Julia code to Python.

Key functions:
- dina_vb(): Variational Bayes for DINA model parameters
- update_Q_item(): Q-vector update via posterior probability
- single_run(): One chain of stochastic VB + Q-update
- dina_Q_estimation_VB(): Full estimation with parallel runs
"""

import numpy as np
from scipy.special import digamma, betaln, gammaln
from scipy.stats import beta as beta_dist
from joblib import Parallel, delayed
from dina_utils import simA, asbinary, reorder_Q, delta, compute_recovery


def dina_vb(X, Q, max_it=1000, epsilon=1e-5):
    """
    Variational Bayes inference for the DINA model.
    
    Parameters
    ----------
    X : (N, J) response matrix
    Q : (J, K) current Q-matrix estimate
    max_it : max VB iterations
    epsilon : convergence threshold
    
    Returns
    -------
    dict with s_est, g_est, att_pat_est, ELBO
    """
    N, J = X.shape
    K = Q.shape[1]
    L = 2 ** K
    A = simA(K)

    # Compute ideal responses for all (item, class) pairs
    natt = Q.sum(axis=1, keepdims=True)  # (J, 1)
    cc = Q @ A.T  # (J, L)
    eta_jl = (cc == natt).astype(np.float64)  # (J, L)

    # Priors
    a_s, b_s = 1.5, 2.0
    a_g, b_g = 1.5, 2.0
    delta_0 = np.ones(L)

    # Initialize
    r_il = np.full((N, L), 1.0 / L)
    onemX = 1.0 - X
    r_il_eta = r_il @ eta_jl.T  # (N, J)

    ELBO = -np.inf

    for m in range(max_it):
        # M-step
        delta_ast = r_il.sum(axis=0) + delta_0
        a_s_ast = (r_il_eta * onemX).sum(axis=0) + a_s
        b_s_ast = (r_il_eta * X).sum(axis=0) + b_s
        a_g_ast = ((1.0 - r_il_eta) * X).sum(axis=0) + a_g
        b_g_ast = ((1.0 - r_il_eta) * onemX).sum(axis=0) + b_g

        # E-step
        E_log_s = digamma(a_s_ast) - digamma(a_s_ast + b_s_ast)
        E_log_1_s = digamma(b_s_ast) - digamma(a_s_ast + b_s_ast)
        E_log_g = digamma(a_g_ast) - digamma(a_g_ast + b_g_ast)
        E_log_1_g = digamma(b_g_ast) - digamma(a_g_ast + b_g_ast)
        E_log_pi = digamma(delta_ast) - digamma(delta_ast.sum())

        lrho = (X * (E_log_1_s - E_log_g)[np.newaxis, :] +
                onemX * (E_log_s - E_log_1_g)[np.newaxis, :]) @ eta_jl + \
               E_log_pi[np.newaxis, :]

        # Normalize (log-sum-exp trick for stability)
        lrho_max = lrho.max(axis=1, keepdims=True)
        tmp = np.exp(lrho - lrho_max)
        r_il = tmp / tmp.sum(axis=1, keepdims=True)
        r_il_eta = r_il @ eta_jl.T

        # ELBO
        ll = np.sum(r_il_eta * (X * E_log_1_s[np.newaxis, :] +
                                 onemX * E_log_s[np.newaxis, :])) + \
             np.sum((1.0 - r_il_eta) * (X * E_log_g[np.newaxis, :] +
                                         onemX * E_log_1_g[np.newaxis, :]))

        kl_pi = np.sum((delta_ast - delta_0) * E_log_pi) - \
                np.sum(gammaln(delta_ast)) + gammaln(delta_ast.sum()) + \
                np.sum(gammaln(delta_0)) - gammaln(delta_0.sum())
        # Note: sign follows Julia code convention

        kl_s = np.sum((a_s_ast - a_s) * digamma(a_s_ast) +
                       (b_s_ast - b_s) * digamma(b_s_ast) -
                       (a_s_ast + b_s_ast - a_s - b_s) * digamma(a_s_ast + b_s_ast) -
                       (betaln(a_s_ast, b_s_ast) - betaln(a_s, b_s)))

        kl_g = np.sum((a_g_ast - a_g) * digamma(a_g_ast) +
                       (b_g_ast - b_g) * digamma(b_g_ast) -
                       (a_g_ast + b_g_ast - a_g - b_g) * digamma(a_g_ast + b_g_ast) -
                       (betaln(a_g_ast, b_g_ast) - betaln(a_g, b_g)))

        r_safe = np.maximum(r_il, 1e-300)
        entropy = -np.sum(r_safe * np.log(r_safe))
        class_ll = np.sum(r_il * E_log_pi[np.newaxis, :])

        new_ELBO = ll + class_ll + entropy - kl_pi - kl_s - kl_g

        if abs(new_ELBO - ELBO) < epsilon:
            ELBO = new_ELBO
            break
        ELBO = new_ELBO

    # Estimated attribute patterns
    att_pat_est = A[r_il.argmax(axis=1), :]

    return {
        "s_est": a_s_ast / (a_s_ast + b_s_ast),
        "g_est": a_g_ast / (a_g_ast + b_g_ast),
        "att_pat_est": att_pat_est,
        "ELBO": ELBO
    }


def update_Q_item(allQ, g_j, s_j, Y_j, alpha, gamma_j):
    """
    Update q-vector for one item via posterior probability.
    
    Parameters
    ----------
    allQ : (H, K) all candidate q-vectors
    g_j, s_j : float — guess and slip for item j
    Y_j : (N,) response vector for item j
    alpha : (N, K) estimated attribute profiles
    gamma_j : (K,) prior probability for each skill
    
    Returns
    -------
    (K,) updated q-vector
    """
    H = allQ.shape[0]
    natt = allQ.sum(axis=1, keepdims=True)  # (H, 1)
    eta_hi = (allQ @ alpha.T == natt).astype(np.float64)  # (H, N)

    g_j = np.clip(g_j, 1e-10, 1.0 - 1e-10)
    s_j = np.clip(s_j, 1e-10, 1.0 - 1e-10)

    Y_mat = Y_j[np.newaxis, :]  # (1, N) broadcast to (H, N)

    ll = np.sum(
        eta_hi * (Y_mat * np.log(1.0 - s_j) + (1.0 - Y_mat) * np.log(s_j)) +
        (1.0 - eta_hi) * (Y_mat * np.log(g_j) + (1.0 - Y_mat) * np.log(1.0 - g_j)),
        axis=1
    )  # (H,)

    gamma_j = np.clip(gamma_j, 1e-10, 1.0 - 1e-10)
    log_prior = allQ @ np.log(gamma_j) + (1.0 - allQ) @ np.log(1.0 - gamma_j)  # (H,)

    pm = ll + log_prior
    pm = np.exp(pm - pm.max())
    pm = pm / pm.sum()

    return allQ[pm.argmax(), :]


def single_run(X, K, niter=550, batchsize=200, seed=1234):
    """
    One chain: iterate VB + Q-update with mini-batches.
    
    Parameters
    ----------
    X : (N, J) response matrix
    K : int — number of attributes
    niter : int — number of outer iterations
    batchsize : int — mini-batch size
    seed : int — random seed
    
    Returns
    -------
    dict with ELBO (niter,) and Qsample (J, K, niter)
    """
    rng = np.random.RandomState(seed)
    N, J = X.shape
    A = simA(K)
    allQ = A[1:, :]  # exclude zero vector
    H = allQ.shape[0]

    # Random initialization
    Q = allQ[rng.choice(H, J, replace=True), :]

    Qsamp = np.zeros((J, K, niter))
    ELBOs = np.zeros(niter)

    for t in range(niter):
        # Mini-batch
        idx = rng.choice(N, min(batchsize, N), replace=True)
        Xs = X[idx, :]

        # VB step
        vb = dina_vb(Xs, Q)

        # Q-update: Beta(1+q, 2-q) posterior mean = (1+q)/3
        gam = (Q + 1.0) / 3.0

        Q_new = np.zeros((J, K))
        for j in range(J):
            Q_new[j, :] = update_Q_item(
                allQ, vb["g_est"][j], vb["s_est"][j],
                Xs[:, j], vb["att_pat_est"], gam[j, :])

        Q = Q_new
        Qsamp[:, :, t] = Q
        ELBOs[t] = vb["ELBO"]

    return {"ELBO": ELBOs, "Qsample": Qsamp}


def dina_Q_estimation_VB(K, X, Qtrue=None, nrun=10, niter=550, nburn=50,
                          batchsize=200, seed=4649, n_jobs=7):
    """
    Full VB Q-matrix estimation with parallel runs.
    
    Parameters
    ----------
    K : int — number of attributes
    X : (N, J) response matrix
    Qtrue : (J, K) true Q-matrix (for recovery computation; None if unknown)
    nrun : int — number of parallel chains
    niter : int — iterations per chain
    nburn : int — burn-in to discard
    batchsize : int — mini-batch size
    seed : int — random seed
    n_jobs : int — parallel workers
    
    Returns
    -------
    dict with recovery, estimated Q, ELBOs
    """
    rng = np.random.RandomState(seed)
    N, J = X.shape
    run_seeds = rng.choice(range(1000, 10000), nrun, replace=False)

    # Parallel runs
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_run)(X, K, niter, batchsize, int(s))
        for s in run_seeds
    )

    # Collect
    ELBOs = np.column_stack([r["ELBO"] for r in results])  # (niter, nrun)
    Qsample = np.stack([r["Qsample"] for r in results], axis=3)  # (J, K, niter, nrun)

    # Best run by mean ELBO after burn-in
    burn_elbos = ELBOs[nburn:, :]
    best_run = np.argmax(burn_elbos.mean(axis=0))

    # Polyak-Ruppert averaging
    Qout = Qsample[:, :, nburn:, best_run]  # (J, K, niter-nburn)
    qest = Qout.mean(axis=2)  # (J, K)

    # Recovery
    if Qtrue is not None:
        recovery = compute_recovery(Qtrue, qest, K)
        C = np.where(Qtrue == 1.0, 0.8, 0.2)
        Qest_final = np.round(reorder_Q(C, qest, K))
    else:
        recovery = None
        Qest_final = np.round(qest)

    return {
        "round_elbomax_recovery": recovery,
        "round_elbomax_Qest": Qest_final,
        "ELBOs": ELBOs,
        "qest_raw": qest
    }


if __name__ == "__main__":
    from dina_utils import get_Q_true_K3J10, simulate_DINA_data
    import time

    Q_true = get_Q_true_K3J10()
    sim = simulate_DINA_data(Q_true, N=500)

    print("Running VB estimation (nrun=3, niter=100 for quick test)...")
    t0 = time.time()
    result = dina_Q_estimation_VB(
        K=3, X=sim["Y"], Qtrue=Q_true,
        nrun=3, niter=100, nburn=20,
        batchsize=300, seed=1234, n_jobs=3)
    elapsed = time.time() - t0

    print(f"Time: {elapsed:.1f}s")
    print(f"Recovery: {result['round_elbomax_recovery']:.4f}")
    print("Estimated Q:")
    print(result["round_elbomax_Qest"])
    print("True Q:")
    print(Q_true)
