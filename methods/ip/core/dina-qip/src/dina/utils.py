# =============================== src/qem_dina/utils.py ========================

import numpy as np

def _qrow_to_mask(qrow: np.ndarray) -> int:
    mask = 0
    for k, val in enumerate(qrow.astype(int).tolist()):
        if val: mask |= (1 << k)
    return mask


def _build_Xi_from_Q(Q: np.ndarray) -> np.ndarray:
    J, K = Q.shape
    P = 1 << K
    Xi = np.zeros((P, J), dtype=np.int8)
    q_masks = [ _qrow_to_mask(Q[j]) for j in range(J) ]
    for a in range(P):
        for j, m in enumerate(q_masks):
            Xi[a, j] = 1 if (a & m) == m else 0
    return Xi


def _superset_zeta_transform(tau: np.ndarray, K: int) -> np.ndarray:
    N, P = tau.shape
    Z = tau.copy()
    for k in range(K):
        step = 1 << k
        for mask in range(1 << K):
            if (mask & step) == 0:
                Z[:, mask] += Z[:, mask | step]
    return Z


def _loglik(R: np.ndarray, s: np.ndarray, g: np.ndarray, nu: np.ndarray, Xi: np.ndarray, clip=1e-12) -> float:
    N, J = R.shape
    P, JJ = Xi.shape
    assert JJ == J
    p = (Xi * (1 - s)[None, :] + (1 - Xi) * g[None, :]).clip(clip, 1 - clip)
    logw = (R[None, :, :] * np.log(p[:, None, :]) + (1 - R)[None, :, :] * np.log(1 - p[:, None, :])).sum(axis=2)
    m = np.max(logw, axis=0)
    return float((m + np.log((nu[:, None] * np.exp(logw - m[None, :])).sum(axis=0))).sum())


def compute_Q_objective(R, s, g, tau, Q, eps=1e-12) -> float:
    s_clip = np.clip(s, eps, 1 - eps)
    g_clip = np.clip(g, eps, 1 - eps)
    beta = R * np.log((1 - s_clip)[None, :] / g_clip[None, :]) + (1 - R) * np.log(s_clip[None, :] / (1 - g_clip)[None, :])
    w = beta.T @ tau  # (J,P)
    Xi = _build_Xi_from_Q(Q)  # (P,J)
    return float((w * Xi.T).sum())





