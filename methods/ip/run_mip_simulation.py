"""
DINA EM-MIP Simulation Study
Runs DINAEM (professor's integer programming method) with multistart=20 and multistart=50
across N=250, 500, 1000 and 100 replications each.

Usage:
    python run_mip_simulation.py

Output:
    results_mip_simulation.csv
    results_mip_simulation.txt  (human-readable table)
"""

import numpy as np
import time
import csv
import sys
import os

from dina_utils import get_Q_true_K3J10, simulate_DINA_data, compute_recovery

# --- import professor's package ---
try:
    from dina.em import DINAEM
    from dina.config import EMConfig, QMipConfig
except ImportError as e:
    print(f"ERROR: Could not import dina-qip package: {e}")
    print("Make sure you ran: pip install -e /path/to/dina-qip")
    sys.exit(1)


# -----------------------------------------------------------------------
# Wrapper: matches the call signature used by run_method_simulation()
# -----------------------------------------------------------------------

def dina_Q_estimation_MIP(K, X, Qtrue=None, multistart=20, seed=4649, **kwargs):
    """
    Thin wrapper around DINAEM.fit() to match the simulation harness signature.
    """
    em_cfg = EMConfig(
        multistart=multistart,
        max_iter=100,
        tol=1e-5,
        verbose=False,
        enforce_one_minus_s_ge_g=True,
    )
    q_cfg = QMipConfig(
        include_identity=True,
        log_to_console=False,
    )

    result = DINAEM().fit(X, K, em_cfg=em_cfg, q_cfg=q_cfg, seed=seed)
    Q_est = result["Q"]

    recovery = compute_recovery(Qtrue, Q_est, K) if Qtrue is not None else None
    return {
        "round_elbomax_recovery": recovery,
        "round_elbomax_Qest": Q_est,
        "loglik": result.get("loglik", None),
    }


# -----------------------------------------------------------------------
# Simulation harness
# -----------------------------------------------------------------------

def run_method_simulation(method_name, estimate_fn, Q_true, N, n_rep=100,
                           seed=4649, **kwargs):
    """
    Run one method across n_rep replications at sample size N.
    Returns eMRR, mMRR, mean time, and per-rep details.
    """
    K = Q_true.shape[1]
    rng = np.random.RandomState(seed)
    rep_seeds = rng.choice(range(1000, 100000), n_rep, replace=False)

    recoveries = []
    times = []

    print(f"\n{'='*60}")
    print(f"  {method_name} — N={N}, {n_rep} replications")
    print(f"{'='*60}")
    sys.stdout.flush()

    for rep in range(n_rep):
        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"  Rep {rep+1}/{n_rep} ...", flush=True)

        np.random.seed(rep_seeds[rep])
        sim = simulate_DINA_data(Q_true, N)

        t0 = time.time()
        result = estimate_fn(
            K=K, X=sim["Y"], Qtrue=Q_true,
            seed=int(rep_seeds[rep]), **kwargs
        )
        elapsed = time.time() - t0

        recoveries.append(result["round_elbomax_recovery"])
        times.append(elapsed)

    eMRR = np.mean(recoveries)
    mMRR = int(np.sum(np.array(recoveries) == 1.0))
    mean_time = np.mean(times)

    print(f"\n  >> eMRR={eMRR*100:.2f}%  mMRR={mMRR}/{n_rep}  mean_time={mean_time:.1f}s")
    sys.stdout.flush()

    return {
        "eMRR": eMRR,
        "mMRR": mMRR,
        "mean_time": mean_time,
        "all_recovery": recoveries,
        "all_times": times,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":

    Q_true = get_Q_true_K3J10()
    K = 3
    N_values = [250, 500, 1000]
    n_rep = 100  # set to 3 for a quick smoke test

    METHODS = [
        ("EM-MIP-20", 20, 7649),
        ("EM-MIP-50", 50, 8649),
    ]

    print("=" * 70)
    print("  DINA EM-MIP Simulation Study")
    print(f"  K={K}, J={Q_true.shape[0]}, rho=0, {n_rep} replications")
    print(f"  Methods: multistart=20 and multistart=50")
    print("=" * 70)
    sys.stdout.flush()

    all_results = {}

    for method_name, ms, base_seed in METHODS:
        for N in N_values:
            key = (method_name, N)
            r = run_method_simulation(
                method_name, dina_Q_estimation_MIP, Q_true, N,
                n_rep=n_rep,
                seed=base_seed + N,
                multistart=ms,
            )
            all_results[key] = r

    # -----------------------------------------------------------------------
    # Print results table
    # -----------------------------------------------------------------------
    header_line = "=" * 75
    table_lines = []
    table_lines.append(header_line)
    table_lines.append(f"  DINA EM-MIP Results — K={K}, J={Q_true.shape[0]}, rho=0, {n_rep} reps")
    table_lines.append(header_line)

    col_header = f"{'Method':<14}"
    for N in N_values:
        col_header += f"  N={N:<4}  {'mMRR':>7} {'eMRR':>7} {'T(s)':>6}"
    table_lines.append(col_header)
    table_lines.append("-" * 75)

    for method_name, ms, _ in METHODS:
        row = f"{method_name:<14}"
        for N in N_values:
            r = all_results[(method_name, N)]
            row += f"         {r['mMRR']:>3}/{n_rep:<3} {r['eMRR']*100:>6.1f}% {r['mean_time']:>5.1f}"
        table_lines.append(row)

    table_lines.append("-" * 75)

    full_table = "\n".join(table_lines)
    print("\n" + full_table)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    # CSV — one row per (method, N, rep)
    csv_path = "results_mip_simulation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "N", "K", "J", "rep", "recovery", "time_s"])
        for (method_name, N), r in all_results.items():
            for rep_idx, (rec, t) in enumerate(zip(r["all_recovery"], r["all_times"])):
                writer.writerow([method_name, N, K, Q_true.shape[0], rep_idx + 1,
                                  f"{rec:.6f}", f"{t:.3f}"])

    # TXT — human-readable summary table
    txt_path = "results_mip_simulation.txt"
    with open(txt_path, "w") as f:
        f.write(full_table + "\n")

    print(f"\nResults saved to {csv_path} and {txt_path}")
