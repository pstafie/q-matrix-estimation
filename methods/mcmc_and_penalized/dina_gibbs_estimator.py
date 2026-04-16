"""
DINA Gibbs Sampler Q-Matrix Estimator
Translation of DINA_QestMCMC.jl (Chung, 2019) to Python.
"""

import numpy as np
from scipy.stats import beta as beta_dist, dirichlet as dirichlet_dist
from itertools import permutations
from joblib import Parallel, delayed
from dina_utils import simA, asbinary, reorder_Q, delta, compute_recovery


def gibbs_single_chain(K, X, A, niter=10000, nburn=5000, seed=1234):
    """
    Single chain of the Gibbs sampler for DINA Q-matrix estimation.
    
    At each iteration:
    1. Sample attribute profiles from conditional posterior
    2. Sample slip parameters
    3. Sample guess parameters (with monotonicity constraint)
    4. Sample mixing proportions
    5. Sample each q-vector from its conditional posterior
    """
    rng = np.random.RandomState(seed)
    N, J = X.shape
    L = 2 ** K
    allQ = A[1:, :]  # exclude zero vector, (H, K)
    H = allQ.shape[0]

    # Initialize Q randomly
    Q = allQ[rng.choice(H, J, replace=True), :]

    # Storage
    Qout_sample = np.zeros((J, K, niter))
    s_sample = np.zeros((niter + 1, J))
    g_sample = np.zeros((niter + 1, J))
    pi_sample = np.zeros((niter + 1, L))

    # Priors
    alpha_s, beta_s_prior = 1.0, 1.0
    alpha_g, beta_g_prior = 1.0, 1.0
    delta_0 = np.ones(L)

    # Initialize parameters
    s_sample[0, :] = rng.uniform(0.1, 0.3, J)
    g_sample[0, :] = rng.uniform(0.1, 0.3, J)
    pi_temp = rng.random(L)
    pi_sample[0, :] = pi_temp / pi_temp.sum()

    onemX = 1.0 - X

    for t in range(niter):
        # Current Q -> eta
        natt = Q.sum(axis=1, keepdims=True)
        cc = Q @ A.T
        eta_jl = (cc == natt).astype(np.float64)
        eta_lj = eta_jl.T  # (L, J)

        # --- Sample latent class indicators z_il ---
        s_t = s_sample[t, :]
        g_t = g_sample[t, :]
        pi_t = pi_sample[t, :]

        log_rho = (X * (np.log(np.clip(1.0 - s_t, 1e-300, None)) -
                        np.log(np.clip(g_t, 1e-300, None)))[np.newaxis, :] +
                   onemX * (np.log(np.clip(s_t, 1e-300, None)) -
                            np.log(np.clip(1.0 - g_t, 1e-300, None)))[np.newaxis, :]) @ eta_lj.T + \
                  np.log(np.clip(pi_t, 1e-300, None))[np.newaxis, :]

        log_rho_max = log_rho.max(axis=1, keepdims=True)
        tmp = np.exp(log_rho - log_rho_max)
        r_il = tmp / tmp.sum(axis=1, keepdims=True)

        # Sample z_il (one-hot per student)
        z_il = np.zeros((N, L))
        for i in range(N):
            chosen = rng.choice(L, p=r_il[i, :])
            z_il[i, chosen] = 1.0

        z_il_eta = z_il @ eta_lj  # (N, J)

        # --- Sample slip ---
        a_sj = (z_il_eta * onemX).sum(axis=0) + alpha_s
        b_sj = (z_il_eta * X).sum(axis=0) + beta_s_prior
        s_sample[t + 1, :] = np.array([rng.beta(a_sj[j], b_sj[j]) for j in range(J)])

        # --- Sample guess (with monotonicity: g < 1-s) ---
        a_gj = ((1.0 - z_il_eta) * X).sum(axis=0) + alpha_g
        b_gj = ((1.0 - z_il_eta) * onemX).sum(axis=0) + beta_g_prior
        for j in range(J):
            upper = 1.0 - s_sample[t + 1, j]
            # Truncated beta: rejection sampling
            for _ in range(100):
                g_candidate = rng.beta(a_gj[j], b_gj[j])
                if g_candidate < upper:
                    g_sample[t + 1, j] = g_candidate
                    break
            else:
                g_sample[t + 1, j] = upper * 0.5

        # --- Sample mixing proportions ---
        delta_ast = z_il.sum(axis=0) + delta_0
        pi_sample[t + 1, :] = rng.dirichlet(delta_ast)

        # --- Sample Q (per item) ---
        s_cur = s_sample[t + 1, :]
        g_cur = g_sample[t + 1, :]
        att_est = A[z_il.argmax(axis=1), :]  # (N, K)

        for j in range(J):
            # Log-likelihood for each candidate q-vector
            natt_h = allQ.sum(axis=1, keepdims=True)
            eta_hi = (allQ @ att_est.T == natt_h).astype(np.float64)  # (H, N)

            s_j = np.clip(s_cur[j], 1e-10, 1 - 1e-10)
            g_j = np.clip(g_cur[j], 1e-10, 1 - 1e-10)

            ll = np.sum(
                eta_hi * (X[:, j][np.newaxis, :] * np.log(1 - s_j) +
                          (1 - X[:, j])[np.newaxis, :] * np.log(s_j)) +
                (1 - eta_hi) * (X[:, j][np.newaxis, :] * np.log(g_j) +
                                (1 - X[:, j])[np.newaxis, :] * np.log(1 - g_j)),
                axis=1)  # (H,)

            # Uniform prior over q-vectors
            pm = np.exp(ll - ll.max())
            pm = pm / pm.sum()

            chosen_h = rng.choice(H, p=pm)
            Q[j, :] = allQ[chosen_h, :]

        Qout_sample[:, :, t] = Q

    return {
        "Qout_sample": Qout_sample,
        "s_sample": s_sample,
        "g_sample": g_sample,
        "pi_sample": pi_sample
    }


def dina_Q_estimation_Gibbs(K, X, Qtrue=None, nchain=3, niter=10000, nburn=5000,
                             seed=4649, n_jobs=3):
    """
    Full Gibbs Q-matrix estimation with multiple chains.
    """
    rng = np.random.RandomState(seed)
    N, J = X.shape
    A = simA(K)
    chain_seeds = rng.choice(range(1000, 10000), nchain, replace=False)

    # Run chains in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(gibbs_single_chain)(K, X, A, niter, nburn, int(s))
        for s in chain_seeds
    )

    # Collect post-burn-in Q samples across all chains
    nmb = niter - nburn
    all_Q = np.zeros((J, K, nchain * nmb))
    for c in range(nchain):
        all_Q[:, :, c * nmb:(c + 1) * nmb] = results[c]["Qout_sample"][:, :, nburn:]

    # Average across all post-burn-in samples
    qest = all_Q.mean(axis=2)

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
        "qest_raw": qest
    }


if __name__ == "__main__":
    from dina_utils import get_Q_true_K3J10, simulate_DINA_data
    import time

    Q_true = get_Q_true_K3J10()
    sim = simulate_DINA_data(Q_true, N=500)

    print("Running Gibbs estimation (nchain=3, niter=2000, nburn=1000)...")
    t0 = time.time()
    result = dina_Q_estimation_Gibbs(
        K=3, X=sim["Y"], Qtrue=Q_true,
        nchain=3, niter=2000, nburn=1000,
        seed=1234, n_jobs=3)
    elapsed = time.time() - t0

    print(f"Time: {elapsed:.1f}s")
    print(f"Recovery: {result['round_elbomax_recovery']:.4f}")
    print("Estimated Q:")
    print(result["round_elbomax_Qest"])
