# src/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, List, Tuple


# -------------------------
# EM
# -------------------------
@dataclass
class EMConfig:
    multistart: int = 1
    max_iter: int = 100
    tol: float = 1e-5
    verbose: bool = True
    enforce_one_minus_s_ge_g: bool = True
    inequality_tol: float = 0.0
    project_how: str = "clip_g"  # or "clip_s"


# -------------------------
# Shared structural constraints for Q
# -------------------------
@dataclass
class QConstraintConfig:
    include_identity: bool = False
    identity_allowed_cols: Optional[Sequence[bool]] = None

    include_distinctness: bool = False
    distinctness_use_indicators: bool = False

    col_lb: Optional[Sequence[Optional[int]]] = None
    col_up: Optional[Sequence[Optional[int]]] = None
    row_lb: Optional[Sequence[Optional[int]]] = None
    row_up: Optional[Sequence[Optional[int]]] = None

    lexi: bool = False
    lexi_ascending: bool = False
    lexi_strict: bool = False
    lexi_row_order: Optional[Sequence[int]] = None

    hierarchy_edges: Optional[List[Tuple[int, int]]] = None
    hierarchy_transitive: bool = True


# -------------------------
# Q estimation (MIP) config
# -------------------------
@dataclass
class QMipConfig(QConstraintConfig):
    time_limit: Optional[float] = None
    mip_gap: Optional[float] = None
    log_to_console: bool = True
    clip_eps: float = 1e-12


# -------------------------
# Q init sampling (SAT/MaxSAT) config
# -------------------------
@dataclass
class QSatConfig(QConstraintConfig):
    """
    Config for sampling Q_init.

    Key principle:
      - If you want lex in init, set lexi=True.
      - If you want to drop lex for diversity, set lexi=False.
      - Same for distinctness, identity, bounds, hierarchy, etc.
    """

    # mixture fractions
    frac_assumption_sat: float = 0.7
    frac_sparse_maxsat: float = 0.3
    frac_dense_maxsat: float = 0.0  # optional

    # assumption-SAT knobs
    assumption_frac: float = 0.25
    tries_per_start: int = 40

    # MaxSAT knobs
    sparse_weight_low: int = 1
    sparse_weight_high: int = 5
    dense_weight_low: int = 1
    dense_weight_high: int = 5

    # diversity control
    dedup: bool = True
    min_hamming_frac: float = 0.05
    max_attempt_factor: float = 3.0

    # solver controls (strings => map inside q_init_sat.py)
    solver_name: str = "g4"
    seed_offset: int = 0

    def __post_init__(self) -> None:
        fr = self.frac_assumption_sat + self.frac_sparse_maxsat + self.frac_dense_maxsat
        if fr <= 0:
            raise ValueError("QSatConfig: at least one frac_* must be > 0.")
        if fr > 1.0 + 1e-12:
            raise ValueError("QSatConfig: frac_assumption_sat+frac_sparse_maxsat+frac_dense_maxsat must be <= 1.")
        if not (0.0 <= self.min_hamming_frac <= 1.0):
            raise ValueError("QSatConfig: min_hamming_frac must be in [0,1].")
        if not (0.0 <= self.assumption_frac <= 1.0):
            raise ValueError("QSatConfig: assumption_frac must be in [0,1].")
        if self.tries_per_start < 1:
            raise ValueError("QSatConfig: tries_per_start must be >= 1.")
        if self.max_attempt_factor < 1.0:
            raise ValueError("QSatConfig: max_attempt_factor must be >= 1.0.")

    @classmethod
    def from_qmip(cls, q: QMipConfig, **overrides) -> "QSatConfig":
        """
        Convenience: start from the *same structural constraints* as the MIP config,
        then override whatever you want explicitly (e.g., set lexi=False for init).
        """
        d = dict(
            include_identity=q.include_identity,
            identity_allowed_cols=q.identity_allowed_cols,
            include_distinctness=q.include_distinctness,
            distinctness_use_indicators=q.distinctness_use_indicators,
            col_lb=q.col_lb,
            col_up=q.col_up,
            row_lb=q.row_lb,
            row_up=q.row_up,
            lexi=q.lexi,
            lexi_ascending=q.lexi_ascending,
            lexi_strict=q.lexi_strict,
            lexi_row_order=q.lexi_row_order,
            hierarchy_edges=q.hierarchy_edges,
            hierarchy_transitive=q.hierarchy_transitive,
        )
        d.update(overrides)
        return cls(**d)


# -------------------------
# Priors
# -------------------------
@dataclass
class Priors:
    a_s: float = 1.0
    b_s: float = 1.0
    a_g: float = 1.0
    b_g: float = 1.0
    dirichlet_nu: Optional[Sequence[float]] = None
