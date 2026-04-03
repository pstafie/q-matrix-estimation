# =============================== src/qem_dina/updates.py ======================

import numpy as np
from typing import Optional, Dict


def mstep_update_nu(tau: np.ndarray, *, dirichlet_prior: Optional[np.ndarray] = None, eps: float = 1e-12):
    N, P = tau.shape
    mass = tau.sum(axis=0)
    if dirichlet_prior is None:
        nu = mass / max(N, 1)
    else:
        alpha = np.asarray(dirichlet_prior, dtype=float)
        numer = mass + (alpha - 1.0)
        denom = N + float(alpha.sum()) - P
        nu = mass / max(N,1) if abs(denom) < eps else numer / denom
    nu = np.clip(nu, eps, 1.0)
    nu = nu / max(nu.sum(), eps)
    return {"nu": nu, "mass": mass}


def mstep_update_sg(
    R: np.ndarray, Q: np.ndarray, tau: np.ndarray,
    *, s_prev: Optional[np.ndarray] = None, g_prev: Optional[np.ndarray] = None,
    priors: Optional[Dict[str, float]] = None, eps: float = 1e-9,
    s_max: Optional[float] = None, g_max: Optional[float] = None,
    enforce_one_minus_s_ge_g: bool = True, inequality_tol: float = 0.0,
    project_how: str = "clip_g",
):
    from .utils import _superset_zeta_transform, _qrow_to_mask
    N, J = R.shape
    JQ, K = Q.shape
    assert JQ == J
    P = tau.shape[1]
    assert (1 << K) == P
    R = R.astype(float)
    Q = (Q > 0).astype(int)

    Z = _superset_zeta_transform(tau, K)
    q_masks = np.fromiter((_qrow_to_mask(Q[j]) for j in range(J)), dtype=np.int64, count=J)
    gamma = Z[:, q_masks]

    A = gamma.sum(axis=0)
    C = (gamma * R).sum(axis=0)
    B = float(N) - A
    D = ((1.0 - gamma) * R).sum(axis=0)

    if priors is None:
        with np.errstate(invalid='ignore', divide='ignore'):
            s = np.where(A > 0, (A - C) / A, np.nan)
            g = np.where(B > 0, D / B, np.nan)
    else:
        a_s = float(priors.get('a_s', 1.0)); b_s = float(priors.get('b_s', 1.0))
        a_g = float(priors.get('a_g', 1.0)); b_g = float(priors.get('b_g', 1.0))
        s = ((A - C) + (a_s - 1.0)) / np.maximum(A + (a_s + b_s - 2.0), 1e-300)
        g = ( D      + (a_g - 1.0)) / np.maximum(B + (a_g + b_g - 2.0), 1e-300)

    if s_prev is None: s_prev = np.full(J, 0.5)
    if g_prev is None: g_prev = np.full(J, 0.5)
    s = np.where(A > 0, s, s_prev)
    g = np.where(B > 0, g, g_prev)

    s = np.clip(s, eps, 1.0 - eps)
    g = np.clip(g, eps, 1.0 - eps)
    if s_max is not None: s = np.minimum(s, float(s_max))
    if g_max is not None: g = np.minimum(g, float(g_max))

    if enforce_one_minus_s_ge_g:
        if project_how == "clip_s":
            s = np.minimum(s, 1.0 - g + float(inequality_tol))
            s = np.clip(s, eps, 1.0 - eps)
        else:
            g = np.minimum(g, 1.0 - s - float(inequality_tol))
            g = np.clip(g, eps, 1.0 - eps)

    return {"s": s, "g": g, "A": A, "B": B, "C": C, "D": D, "gamma": gamma}

