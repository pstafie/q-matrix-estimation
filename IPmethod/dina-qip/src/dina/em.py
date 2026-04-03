# =============================== src/dina/em.py
# ===============================

from __future__ import annotations

import numpy as np
from typing import Any, Optional, List

from .config import EMConfig, QMipConfig, QSatConfig, Priors
from .estep import compute_tau
from .updates import mstep_update_nu, mstep_update_sg
from .q_mip import mstep_update_Q
from .utils import _build_Xi_from_Q, _loglik, compute_Q_objective
from .q_init_sat import sample_random_Q_inits_with_sat


def _hamming(Qa: np.ndarray, Qb: np.ndarray) -> int:
    Qa = (Qa > 0).astype(np.uint8)
    Qb = (Qb > 0).astype(np.uint8)
    return int(np.count_nonzero(Qa ^ Qb))


def _derive_q_sat_cfg_from_q_cfg(q_cfg: QMipConfig) -> QSatConfig:
    """
    Convenience: if user didn't pass q_sat_cfg, inherit structural constraints
    from q_cfg (identity/distinctness/bounds/lex/hierarchy). Sampling-specific
    knobs remain at QSatConfig defaults.
    """
    fields = [
        "include_identity",
        "identity_allowed_cols",
        "include_distinctness",
        "distinctness_use_indicators",
        "col_lb",
        "col_up",
        "row_lb",
        "row_up",
        "lexi",
        "lexi_ascending",
        "lexi_strict",
        "lexi_row_order",
        "hierarchy_edges",
        "hierarchy_transitive",
    ]
    kwargs = {f: getattr(q_cfg, f) for f in fields if hasattr(q_cfg, f)}
    try:
        return QSatConfig(**kwargs)  # type: ignore[arg-type]
    except TypeError:
        # if your QSatConfig signature differs, fall back to defaults
        cfg = QSatConfig()
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


class DINAEM:
    def __init__(self):
        pass

    # ---------------- internal: run a single start ----------------
    def _run_single_start(
        self,
        R: np.ndarray,
        K: int,
        start_seed: int,
        *,
        em_cfg: EMConfig,
        q_cfg: QMipConfig,
        priors: Optional[Priors],
        Q0: np.ndarray,
        s_init: Optional[np.ndarray],
        g_init: Optional[np.ndarray],
        nu_init: Optional[np.ndarray],
    ) -> dict[str, Any]:
        N, J = R.shape
        P = 1 << K

        Q = (Q0 > 0).astype(int)

        rng = np.random.default_rng(start_seed)

        s = np.array(
            s_init if s_init is not None else np.clip(0.2 + rng.normal(0, 0.05, size=J), 1e-3, 0.95),
            float,
        )
        g = np.array(
            g_init if g_init is not None else np.clip(0.2 + rng.normal(0, 0.05, size=J), 1e-3, 0.95),
            float,
        )

        nu = np.array(nu_init if nu_init is not None else np.full(P, 1.0 / P), float)
        nu = np.maximum(nu, 1e-12)
        nu /= nu.sum()

        Xi = _build_Xi_from_Q(Q)
        history: List[dict[str, Any]] = []

        ll = _loglik(R, s, g, nu, Xi)
        best_ll = ll
        best = {"Q": Q.copy(), "s": s.copy(), "g": g.copy(), "nu": nu.copy(), "Xi": Xi.copy()}
        history.append({"iter": 0, "loglik": float(ll)})

        if em_cfg.verbose:
            print(f"[EM] init: loglik={ll:.6f}")

        prev_ll = ll
        for it in range(1, em_cfg.max_iter + 1):
            # E-step
            tau = compute_tau(R, s, g, Xi, nu, eps=1e-12)

            # M-step: nu
            nu = mstep_update_nu(tau, dirichlet_prior=(priors.dirichlet_nu if priors else None))["nu"]

            # M-step: Q  (accept only if Q-objective doesn't decrease)
            Q_old = Q.copy()
            phi_old = compute_Q_objective(R, s, g, tau, Q_old)

            out_Q = mstep_update_Q(R, s, g, nu, tau, Q_start=Q_old, **q_cfg.__dict__)
            Q_cand = out_Q["Q"]
            phi_new = compute_Q_objective(R, s, g, tau, Q_cand)

            if phi_new + 1e-9 < phi_old:
                Q = Q_old
            else:
                Q = Q_cand

            Xi = _build_Xi_from_Q(Q)

            # M-step: s,g
            out_sg = mstep_update_sg(
                R,
                Q,
                tau,
                s_prev=s,
                g_prev=g,
                priors=(vars(priors) if priors else None),
                eps=1e-9,
                enforce_one_minus_s_ge_g=em_cfg.enforce_one_minus_s_ge_g,
                inequality_tol=em_cfg.inequality_tol,
                project_how=em_cfg.project_how,
            )
            s, g = out_sg["s"], out_sg["g"]

            # observed loglik
            ll = _loglik(R, s, g, nu, Xi)
            history.append({"iter": it, "loglik": float(ll)})

            rel = abs(ll - prev_ll) / max(1.0, abs(prev_ll))
            if em_cfg.verbose:
                print(f"[EM] iter {it:03d}: loglik={ll:.6f}  Δ={ll - prev_ll:.6e}  rel={rel:.3e}")

            if ll > best_ll + 1e-12:
                best_ll = ll
                best = {"Q": Q.copy(), "s": s.copy(), "g": g.copy(), "nu": nu.copy(), "Xi": Xi.copy()}

            if rel <= em_cfg.tol:
                break
            prev_ll = ll

        return {**best, "loglik": float(best_ll), "history": history}

    # ---------------- public: multistart wrapper ----------------
    def fit(
        self,
        R: np.ndarray,
        K: int,
        *,
        em_cfg: Optional[EMConfig] = None,
        q_cfg: Optional[QMipConfig] = None,
        q_sat_cfg: Optional[QSatConfig] = None,
        priors: Optional[Priors] = None,
        Q_init: Optional[np.ndarray] = None,
        s_init: Optional[np.ndarray] = None,
        g_init: Optional[np.ndarray] = None,
        nu_init: Optional[np.ndarray] = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        base_seed = int(seed)

        if em_cfg is None:
            em_cfg = EMConfig()
        if q_cfg is None:
            q_cfg = QMipConfig()

        if q_sat_cfg is None:
            q_sat_cfg = _derive_q_sat_cfg_from_q_cfg(q_cfg)

        N, J = R.shape
        mult = max(1, int(getattr(em_cfg, "multistart", 1)))

        # global min-Hamming threshold (applies ALSO vs user Q_init)
        min_hamming_frac = float(getattr(q_sat_cfg, "min_hamming_frac", 0.0))
        min_hamming = int(np.ceil(min_hamming_frac * (J * K))) if min_hamming_frac > 0 else 0

        # ---- build the pool of Q starts ----
        Q_starts: List[np.ndarray] = []

        if Q_init is not None:
            user_Q = (Q_init > 0).astype(np.uint8)
            Q_starts.append(user_Q)

            extra = mult - 1
            if extra > 0:
                try:
                    more = sample_random_Q_inits_with_sat(
                        J,
                        K,
                        q_sat_cfg,
                        n_starts=extra,
                        seed=base_seed + 12345,
                    )
                    # enforce: not equal + min-Hamming vs user_Q
                    for Qc in more:
                        if len(Q_starts) >= mult:
                            break
                        Qc_u = (Qc > 0).astype(np.uint8)
                        if np.array_equal(Qc_u, user_Q):
                            continue
                        if min_hamming > 0 and _hamming(Qc_u, user_Q) < min_hamming:
                            continue
                        Q_starts.append(Qc_u)
                except Exception as e:
                    if em_cfg.verbose:
                        print(f"[EM] Q-init sampling failed ({e}); proceeding with user Q_init only.")
        else:
            try:
                Q_starts = sample_random_Q_inits_with_sat(
                    J,
                    K,
                    q_sat_cfg,
                    n_starts=mult,
                    seed=base_seed + 54321,
                )
                if not Q_starts:
                    raise RuntimeError("No SAT starts found.")
                if em_cfg.verbose:
                    print(f"[EM] Prepared {len(Q_starts)} Q starts via mixed SAT/MaxSAT sampler.")
            except Exception as e:
                if em_cfg.verbose:
                    print(f"[EM] Q-init sampling failed ({e}); falling back to simple random init.")
                rng = np.random.default_rng(base_seed + 999)
                Q = np.zeros((J, K), dtype=np.uint8)
                for k in range(min(J, K)):
                    Q[k, k] = 1
                if J > K:
                    Q[K:, :] = rng.integers(0, 2, size=(J - K, K), dtype=np.uint8)
                Q_starts = [Q]

        if em_cfg.verbose:
            print(f"[EM] Multistart requested: {mult}, prepared starts: {len(Q_starts)}")

        # ---- run each start and pick the best observed log-likelihood ----
        best_run: Optional[dict[str, Any]] = None
        for idx, Q0 in enumerate(Q_starts, start=1):
            if em_cfg.verbose and len(Q_starts) > 1:
                print(f"\n[EM] === Multistart {idx}/{len(Q_starts)} ===")

            start_seed = base_seed + idx - 1
            out = self._run_single_start(
                R,
                K,
                start_seed=start_seed,
                em_cfg=em_cfg,
                q_cfg=q_cfg,
                priors=priors,
                Q0=Q0,
                s_init=s_init,
                g_init=g_init,
                nu_init=nu_init,
            )
            if (best_run is None) or (out["loglik"] > best_run["loglik"]):
                best_run = out

        assert best_run is not None
        return best_run
