import numpy as np
from typing import Dict

def q_entry_accuracy(Q_true: np.ndarray, Q_hat: np.ndarray) -> float:
    return float((Q_true.astype(bool) == Q_hat.astype(bool)).mean())

def q_hamming(Q_true: np.ndarray, Q_hat: np.ndarray) -> int:
    return int(np.count_nonzero(Q_true.astype(bool) ^ Q_hat.astype(bool)))

def q_exact(Q_true: np.ndarray, Q_hat: np.ndarray) -> int:
    return int(np.array_equal(Q_true, Q_hat))

def q_row_recovery(Q_true: np.ndarray, Q_hat: np.ndarray) -> float:
    return float((Q_true.astype(bool) == Q_hat.astype(bool)).all(axis=1).mean())

def sg_rmse(s_true: np.ndarray, g_true: np.ndarray, s_hat: np.ndarray, g_hat: np.ndarray) -> Dict[str, float]:
    return {
        "rmse_s": float(np.sqrt(np.mean((s_true - s_hat)**2))),
        "rmse_g": float(np.sqrt(np.mean((g_true - g_hat)**2))),
    }
