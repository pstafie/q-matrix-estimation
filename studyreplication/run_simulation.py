"""
DINA Q-Matrix Simulation Study — Main Runner
Compares VB, Gibbs, and EM+Lasso on K=3, J=10, rho=0

Usage:
    python run_simulation.py

Output: Comparison table matching the Julia results.
"""

import numpy as np
import time
import csv
from dina_utils import get_Q_true_K3J10, simulate_DINA_data, compute_recovery
from dina_vb_estimator import dina_Q_estimation_VB
from dina_gibbs_estimator import dina_Q_estimation_Gibbs
from dina_lasso_estimator import dina_Q_estimation_Lasso


def run_method_simulation(method_name, estimate_fn, Q_true, N, n_rep=100,
                           seed=4649, **kwargs):
    """
    Run one method across n_rep replications.
    
    Parameters
    ----------
    method_name : str
    estimate_fn : callable — takes (K, X, Qtrue, **kwargs) and returns dict with 'round_elbomax_recovery'
    Q_true : (J, K) true Q-matrix
    N : int — sample size
    n_rep : int — number of replications
    seed : int
    **kwargs : passed to estimate_fn
    
    Returns
    -------
    dict with eMRR, mMRR, mean_time, all_recovery, all_times
    """
    K = Q_true.shape[1]
    rng = np.random.RandomState(seed)
    rep_seeds = rng.choice(range(1000, 100000), n_rep, replace=False)

    recoveries = []
    times = []

    print(f"\n{'='*60}")
    print(f"  {method_name} — N={N}, {n_rep} replications")
    print(f"{'='*60}")

    for rep in range(n_rep):
        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"  Rep {rep+1}/{n_rep}")

        # Generate data
        np.random.seed(rep_seeds[rep])
        sim = simulate_DINA_data(Q_true, N)

        # Estimate
        t0 = time.time()
        result = estimate_fn(K=K, X=sim["Y"], Qtrue=Q_true,
                              seed=int(rep_seeds[rep]), **kwargs)
        elapsed = time.time() - t0

        recoveries.append(result["round_elbomax_recovery"])
        times.append(elapsed)

    eMRR = np.mean(recoveries)
    mMRR = np.sum(np.array(recoveries) == 1.0)
    mean_time = np.mean(times)

    print(f"\n  Results: eMRR={eMRR*100:.2f}%  mMRR={mMRR}/{n_rep}  time={mean_time:.1f}s")

    return {
        "eMRR": eMRR, "mMRR": mMRR, "mean_time": mean_time,
        "all_recovery": recoveries, "all_times": times
    }


if __name__ == "__main__":
    Q_true = get_Q_true_K3J10()
    K = 3
    N_values = [250, 500, 1000]
    n_rep = 100  # Set to 10 for a quick test, 100 for full results

    print("="*70)
    print("  DINA Q-Matrix Simulation Study")
    print(f"  K={K}, J={Q_true.shape[0]}, rho=0, {n_rep} replications")
    print("="*70)

    all_results = {}

    # ---- VB ----
    for N in N_values:
        bs = 200 if N == 250 else 300
        r = run_method_simulation(
            "VB", dina_Q_estimation_VB, Q_true, N, n_rep=n_rep,
            seed=4649 + N,
            nrun=10, niter=550, nburn=50, batchsize=bs, n_jobs=7)
        all_results[("VB", N)] = r

    # ---- Gibbs ----
    for N in N_values:
        r = run_method_simulation(
            "Gibbs", dina_Q_estimation_Gibbs, Q_true, N, n_rep=n_rep,
            seed=5649 + N,
            nchain=3, niter=10000, nburn=5000, n_jobs=3)
        all_results[("Gibbs", N)] = r

    # ---- EM+Lasso ----
    for N in N_values:
        r = run_method_simulation(
            "EM+Lasso", dina_Q_estimation_Lasso, Q_true, N, n_rep=n_rep,
            seed=6649 + N,
            nrun=10, n_jobs=7)
        all_results[("EM+Lasso", N)] = r

    # ---- Print comparison table ----
    print("\n" + "="*75)
    print(f"  Table: Q-Matrix Recovery Rates — K={K}, J={Q_true.shape[0]}, rho=0, {n_rep} reps")
    print("="*75)
    print()
    header = f"{'Method':<12}"
    for N in N_values:
        header += f"  {'mMRR':>7} {'eMRR':>7} {'T(s)':>6}"
    print(header)
    print("-"*75)

    for method in ["VB", "Gibbs", "EM+Lasso"]:
        row = f"{method:<12}"
        for N in N_values:
            r = all_results[(method, N)]
            row += f"  {int(r['mMRR']):>3}/{n_rep:<3} {r['eMRR']*100:>6.1f}% {r['mean_time']:>5.1f}"
        print(row)
    print("-"*75)
    print()
    print("Paper (VB): 41/100 97.0%  —     80/100 99.2%  —     91/100 99.7%  —")

    # ---- Save to CSV ----
    with open("simulation_results_python.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "N", "K", "J", "rho", "mMRR", "eMRR", "mean_time"])
        for (method, N), r in all_results.items():
            writer.writerow([method, N, K, Q_true.shape[0], 0.0,
                             int(r["mMRR"]), f"{r['eMRR']:.6f}", f"{r['mean_time']:.2f}"])
    print("Results saved to simulation_results_python.csv")
