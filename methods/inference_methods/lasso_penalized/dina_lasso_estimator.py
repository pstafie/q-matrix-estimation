"""
DINA EM+Lasso Q-Matrix Estimator
Translation of DINA_Lasso.jl (Chen et al., 2015) to Python.
"""

import numpy as np
from joblib import Parallel, delayed
from dina_utils import simA, asbinary, reorder_Q, delta, compute_recovery


def interact(K):
    """Generate interaction matrix Z (2^K x 2^K)."""
    L = 2 ** K
    A = np.zeros((L, K))
    for m in range(1, L):
        A[m, :] = asbinary(m, K)
    A = A[:, ::-1]  # reverse columns
    X_mat = A.T
    Y_mat = X_mat + 1.0
    Z = np.zeros((L, L))
    for l in range(L):
        Z[:, l] = (np.prod(Y_mat - X_mat[:, l:l+1], axis=0) > 0).astype(float)
    return Z


def logit(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1.0 - p))


def soft_threshold(x, mu):
    return np.sign(x) * np.maximum(np.abs(x) - mu, 0.0)


def prob_func(theta, Z):
    tmp = theta @ Z.T
    return np.exp(tmp) / (1.0 + np.exp(tmp))


def post_func(theta, mixprop, Z, data):
    prob = prob_func(theta, Z)
    logpost = (data @ np.log(prob + 1e-300) +
               (1.0 - data) @ np.log(1.0 - prob + 1e-300)) + \
              np.log(mixprop + 1e-300)[np.newaxis, :]
    logpost_max = logpost.max(axis=1, keepdims=True)
    tmp = np.exp(logpost - logpost_max)
    return tmp / tmp.sum(axis=1, keepdims=True)


def log_lik(theta, mixprop, Z, data):
    prob = prob_func(theta, Z)
    tmp = np.exp(data @ np.log(prob + 1e-300) +
                 (1.0 - data) @ np.log(1.0 - prob + 1e-300)) @ mixprop[:, np.newaxis]
    return np.sum(np.log(tmp + 1e-300))


def obj_func(theta, mixprop, Z, data, lam):
    N = data.shape[0]
    return -log_lik(theta, mixprop, Z, data) / N + np.sum(lam[:, np.newaxis] * np.abs(theta[:, 1:]))


def Mstep_mixprop(post, delta_param):
    tmp = post.sum(axis=0)
    return (tmp + delta_param) / (tmp + delta_param).sum()


def grad_j(post, theta_j, Z, data_j):
    N = len(data_j)
    tmp = (Z @ theta_j).T  # (1, L)
    return (-(data_j @ post @ Z - (post.sum(axis=0) * (np.exp(tmp) / (1.0 + np.exp(tmp)))) @ Z) / N).flatten()


def obj_j(post, theta_j, Z, data_j, lambda_j):
    N = len(data_j)
    tmp = (Z @ theta_j).T
    val = data_j @ post @ tmp.T - np.log(1.0 + np.exp(tmp)) @ post.sum(axis=0)[:, np.newaxis]
    return (-val.item() / N) + lambda_j * np.sum(np.abs(theta_j[1:]))


def Mstep_theta_j(post, theta_j, Z, data_j, lambda_j, step, totalstep):
    g = grad_j(post, theta_j, Z, data_j)
    obj0 = obj_j(post, theta_j, Z, data_j, lambda_j)

    y = theta_j - step * g
    theta_new = np.copy(y)
    theta_new[1:] = soft_threshold(y[1:], lambda_j * step)

    obj1 = obj_j(post, theta_new, Z, data_j, lambda_j)

    for z in range(totalstep):
        if obj1 <= obj0:
            break
        step = step * 0.5
        y = theta_j - step * g
        theta_new[0] = y[0]
        theta_new[1:] = soft_threshold(y[1:], lambda_j * step)
        obj1 = obj_j(post, theta_new, Z, data_j, lambda_j)

    return theta_new


def EM_step(theta, mixprop, Z, data, lam, delta_param=1, step=1, totalstep=20, tol=1e-6):
    N, J = data.shape
    obj0 = obj_func(theta, mixprop, Z, data, lam)

    for _ in range(500):
        post = post_func(theta, mixprop, Z, data)
        mixprop = Mstep_mixprop(post, delta_param)

        hat_theta = theta.copy()
        for j in range(J):
            hat_theta[j, :] = Mstep_theta_j(post, theta[j, :], Z, data[:, j], lam[j], step, totalstep)
        theta = hat_theta

        obj1 = obj_func(theta, mixprop, Z, data, lam)
        if (obj0 - obj1) < tol:
            break
        obj0 = obj1

    return {"hat_theta": theta, "hat_mixprop": mixprop, "obj_value": obj1}


def Path(theta, mixprop, Z, data, lam, A,
         h_thre=0.1, path_total=30, lambda_step=0.01):
    """Run EM with adaptive lambda path."""
    for t in range(path_total):
        result = EM_step(theta, mixprop, Z, data, lam)

        ind_matr = np.abs(result["hat_theta"][:, 1:]) > h_thre
        ind = ind_matr.sum(axis=1)

        if np.sum(ind > 1) == 0:
            break
        else:
            lam[ind > 1] += lambda_step
            theta = result["hat_theta"]
            mixprop = result["hat_mixprop"]

    # Extract Q from theta
    qindex = np.argmax(np.abs(result["hat_theta"][:, 1:]), axis=1) + 1
    zero_items = np.sum(np.abs(result["hat_theta"][:, 1:]), axis=1) < 1e-6
    qindex[zero_items] = 0
    Q = A[qindex, :]

    return {"Q": Q, "obj_value": result["obj_value"], "lambda": lam}


def _run_one_path(c, theta_array, mixprop_array, Z, data, lambda_array, A):
    """Helper for parallel path runs."""
    return Path(theta_array[:, :, c].copy(),
                mixprop_array[c, :].copy(),
                Z, data,
                lambda_array[c, :].copy(), A)


def dina_Q_estimation_Lasso(K, X, Qtrue=None, nrun=10, seed=4649, n_jobs=7):
    """
    Full EM+Lasso Q-matrix estimation.
    """
    rng = np.random.RandomState(seed)
    N, J = X.shape
    L = 2 ** K
    A = simA(K)
    Z = interact(K)

    # Initialize
    theta_array = rng.uniform(0, 1, (J, L, nrun))
    mixprop_array = np.full((nrun, L), 1.0 / L)
    lambda_values = np.linspace(0.01, 0.1, nrun)
    lambda_array = np.zeros((nrun, J))
    for c in range(nrun):
        lambda_array[c, :] = lambda_values[c]

    # Run paths in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_path)(c, theta_array, mixprop_array, Z, X, lambda_array, A)
        for c in range(nrun)
    )

    # Select best by objective value
    obj_values = np.array([r["obj_value"] for r in results])
    Qs = np.stack([r["Q"] for r in results], axis=2)

    # Pick the run with smallest objective that has no zero q-vectors
    sorted_idx = np.argsort(obj_values)
    Q_final = Qs[:, :, sorted_idx[0]]
    for idx in sorted_idx:
        if np.all(Qs[:, :, idx].sum(axis=1) > 0):
            Q_final = Qs[:, :, idx]
            break

    # Recovery
    if Qtrue is not None:
        recovery = compute_recovery(Qtrue, Q_final, K)
        C = np.where(Qtrue == 1.0, 0.8, 0.2)
        Qest_final = np.round(reorder_Q(C, Q_final, K))
    else:
        recovery = None
        Qest_final = np.round(Q_final)

    return {
        "round_elbomax_recovery": recovery,
        "round_elbomax_Qest": Qest_final,
        "Q_final_raw": Q_final
    }


if __name__ == "__main__":
    from dina_utils import get_Q_true_K3J10, simulate_DINA_data
    import time

    Q_true = get_Q_true_K3J10()
    sim = simulate_DINA_data(Q_true, N=500)

    print("Running EM+Lasso estimation (nrun=5 for quick test)...")
    t0 = time.time()
    result = dina_Q_estimation_Lasso(
        K=3, X=sim["Y"], Qtrue=Q_true,
        nrun=5, seed=1234, n_jobs=3)
    elapsed = time.time() - t0

    print(f"Time: {elapsed:.1f}s")
    print(f"Recovery: {result['round_elbomax_recovery']:.4f}")
    print("Estimated Q:")
    print(result["round_elbomax_Qest"])
