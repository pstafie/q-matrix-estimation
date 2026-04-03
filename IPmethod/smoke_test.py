"""
smoke_test.py — run this first on Great Lakes to verify everything works
before submitting the full 100-rep job.

Runs 2 replications at N=250 only, multistart=1, should finish in ~1-2 minutes.

Usage:
    python smoke_test.py
"""

import numpy as np
import sys

print("--- Importing packages ---", flush=True)

try:
    from dina.em import DINAEM
    from dina.config import EMConfig, QMipConfig
    print("  dina-qip: OK", flush=True)
except ImportError as e:
    print(f"  dina-qip FAILED: {e}")
    sys.exit(1)

try:
    from dina_utils import get_Q_true_K3J10, simulate_DINA_data, compute_recovery
    print("  dina_utils: OK", flush=True)
except ImportError as e:
    print(f"  dina_utils FAILED: {e}")
    sys.exit(1)

print("\n--- Running smoke test (2 reps, N=250, multistart=1) ---", flush=True)

Q_true = get_Q_true_K3J10()
np.random.seed(42)
sim = simulate_DINA_data(Q_true, N=250)

em_cfg = EMConfig(multistart=1, max_iter=30, verbose=True)
q_cfg  = QMipConfig(include_identity=True, log_to_console=False)

result = DINAEM().fit(sim["Y"], K=3, em_cfg=em_cfg, q_cfg=q_cfg, seed=0)

recovery = compute_recovery(Q_true, result["Q"], K=3)

print(f"\nEstimated Q:\n{result['Q']}")
print(f"True Q:\n{Q_true.astype(int)}")
print(f"Loglik: {result['loglik']:.4f}")
print(f"Recovery: {recovery:.4f}")
print("\n--- Smoke test PASSED ---")
