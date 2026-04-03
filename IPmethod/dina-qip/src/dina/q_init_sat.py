# src/qem_dina/q_init_sat.py
from __future__ import annotations

from typing import Optional, Sequence, List, Tuple, Dict, Any
import copy
import math
import numpy as np

from pysat.formula import CNF, IDPool, WCNF
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver
from pysat.examples.rc2 import RC2

from .config import QSatConfig
from .SAT_constraints import _as_bounds_vec, _xor_equiv, _apply_hierarchy, add_lex_chain_columns


# -------------------------
# helpers
# -------------------------


def _hamming(Qa: np.ndarray, Qb: np.ndarray) -> int:
    # both uint8 {0,1}
    return int(np.count_nonzero(Qa ^ Qb))


def _alloc_counts(n: int, a: float, b: float, c: float) -> Tuple[int, int, int]:
    """
    Allocate n into (na, nb, nc) proportional-ish to (a,b,c), with sum exactly n.
    """
    w = np.array([a, b, c], float)
    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        return n, 0, 0
    w = w / w.sum()
    raw = w * n
    base = np.floor(raw).astype(int)
    rem = n - int(base.sum())
    # distribute remainder by largest fractional parts
    frac = raw - base
    order = np.argsort(-frac)
    for i in range(rem):
        base[order[i % 3]] += 1
    return int(base[0]), int(base[1]), int(base[2])


def _dedup_and_min_hamming(
    candidates: List[np.ndarray],
    *,
    n_want: int,
    min_hamming: int,
    dedup: bool,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    seen = set()

    for Q in candidates:
        Q = np.asarray(Q, dtype=np.uint8)
        key = Q.tobytes()
        if dedup and key in seen:
            continue

        ok = True
        if min_hamming > 0 and out:
            for Q_prev in out:
                if _hamming(Q, Q_prev) < min_hamming:
                    ok = False
                    break

        if not ok:
            continue

        out.append(Q)
        if dedup:
            seen.add(key)
        if len(out) >= n_want:
            break

    return out


def _extract_fixed_xvars(cnf: CNF, x_vars_set: set[int]) -> Dict[int, int]:
    """
    Identify x-vars forced by unit clauses.
    Returns dict var -> {0,1}.
    """
    fixed: Dict[int, int] = {}
    for cl in cnf.clauses:
        if len(cl) == 1:
            lit = cl[0]
            v = abs(lit)
            if v in x_vars_set:
                fixed[v] = 1 if lit > 0 else 0
    return fixed


def _model_to_Q(model: set[int], var_of, J: int, K: int) -> np.ndarray:
    Q = np.zeros((J, K), dtype=np.uint8)
    for r in range(J):
        for c in range(K):
            v = var_of(r, c)
            Q[r, c] = 1 if v in model else 0
    return Q


# -------------------------
# main CNF builder (reads QSatConfig structural fields)
# -------------------------

def build_Q_sat_from_cfg(
    J: int,
    K: int,
    *,
    include_identity: bool = False,
    identity_allowed_cols: Optional[Sequence[bool]] = None,
    include_distinctness: bool = False,
    distinctness_use_indicators: bool = False,  # kept for API compatibility (unused)
    col_lb: Optional[Sequence[Optional[int]]] = None,
    col_up: Optional[Sequence[Optional[int]]] = None,
    row_lb: Optional[Sequence[Optional[int]]] = None,
    row_up: Optional[Sequence[Optional[int]]] = None,
    lexi: bool = False,
    lexi_ascending: bool = False,
    lexi_strict: bool = False,
    lexi_row_order: Optional[Sequence[int]] = None,
    hierarchy_edges: Optional[List[Tuple[int, int]]] = None,
    hierarchy_transitive: bool = False,
) -> tuple[CNF, IDPool, Dict[str, Any]]:
    if J <= 0 or K <= 0:
        raise ValueError("J and K must be positive integers.")

    # Always enforce that Q contains a K x K identity submatrix.
    # (We keep the include_identity argument for API compatibility,
    #  but it is no longer optional here.)
    if K > J:
        raise ValueError(f"Not enough rows to host identity rows: need {K}, have {J}.")
    id_cols_list = list(range(K))   # all columns participate in the identity

    vpool = IDPool()
    cnf = CNF()

    def var_of(j: int, k: int) -> int:
        return vpool.id(("x", j, k))


    # 0) Pin identity rows at top (for all columns)
    # id_cols_list = list(range(K))
    # for i, k in enumerate(id_cols_list):
    #     r = i
    #     cnf.append([var_of(r, k)])
    #     for kk in range(K):
    #         if kk != k:
    #             cnf.append([-var_of(r, kk)])

    # 0) Pin identity rows anywhere. 
    def y_of(j: int, k: int) -> int:
        # selector: row j is the identity row for column k
        return vpool.id(("y", j, k))

    def add_at_most_one(lits: list[int]) -> None:
        # pairwise AMO (fine for small J,K)
        for a in range(len(lits)):
            for b in range(a + 1, len(lits)):
                cnf.append([-lits[a], -lits[b]])

    # (A) For each required identity column k: choose >=1 row j as its identity row
    for k in id_cols_list:
        cnf.append([y_of(j, k) for j in range(J)])  # at least one

        # OPTIONAL: enforce EXACTLY ONE identity row per column k
        # (keeps things cleaner; comment out if you truly want ">=1")
        add_at_most_one([y_of(j, k) for j in range(J)])

    # (B) OPTIONAL: a row can serve as identity for at most one column
    # (recommended if you used EXACTLY ONE per column above)
    for j in range(J):
        add_at_most_one([y_of(j, k) for k in id_cols_list])

    # (C) Link y[j,k] to the x-variables: y[j,k] -> row j equals e_k
    for j in range(J):
        for k in id_cols_list:
            # y -> x_{j,k} = 1
            cnf.append([-y_of(j, k), var_of(j, k)])
            # y -> x_{j,kk} = 0 for kk != k
            for kk in range(K):
                if kk != k:
                    cnf.append([-y_of(j, k), -var_of(j, kk)])
                

                
    # bounds
    col_lb_vec = _as_bounds_vec(col_lb, K, None)
    col_up_vec = _as_bounds_vec(col_up, K, None)
    row_lb_vec = _as_bounds_vec(row_lb, J, None)
    row_up_vec = _as_bounds_vec(row_up, J, None)

    # 1) row cardinalities
    for r in range(J):
        lits = [var_of(r, c) for c in range(K)]
        lb = row_lb_vec[r]
        ub = row_up_vec[r]
        if lb is not None and lb >= 0:
            cnf.extend(CardEnc.atleast(lits=lits, bound=int(lb), vpool=vpool, encoding=EncType.seqcounter).clauses)
        if ub is not None and ub >= 0:
            cnf.extend(CardEnc.atmost(lits=lits, bound=int(ub), vpool=vpool, encoding=EncType.seqcounter).clauses)

    # 2) column cardinalities
    for c in range(K):
        lits = [var_of(r, c) for r in range(J)]
        lb = col_lb_vec[c]
        ub = col_up_vec[c]
        if lb is not None and lb >= 0:
            cnf.extend(CardEnc.atleast(lits=lits, bound=lb, vpool=vpool, encoding=EncType.seqcounter).clauses)
        if ub is not None and ub >= 0:
            cnf.extend(CardEnc.atmost(lits=lits, bound=ub, vpool=vpool, encoding=EncType.seqcounter).clauses)

    # 3) distinctness across columns ignoring identity rows.
    #    We want columns to differ on at least one *non-identity* row.
    rows_for_distinctness = range(J)
    if include_distinctness:
        if J - K == 0 and K > 1:
            cnf.append([])  # UNSAT: no non-identity rows available
        for c1 in range(K):
            for c2 in range(c1 + 1, K):
                y_lits = []
                for r in rows_for_distinctness:
                    # z is the true XOR at row r
                    z = vpool.id(("z_xor", r, c1, c2))
                    _xor_equiv(cnf, z, var_of(r, c1), var_of(r, c2))

                    # w is a "witness" that is allowed ONLY on non-identity rows
                    w = vpool.id(("w_witness", r, c1, c2))

                    # w -> z  (if it's a witness, it must be a real difference)
                    cnf.append([-w, z])

                    # w -> (row r is not an identity row for ANY column)
                    for k in id_cols_list:
                        cnf.append([-w, -y_of(r, k)])

                    y_lits.append(w)

                # must have at least one non-identity witness row where columns differ
                cnf.append(y_lits)


    # 4) hierarchy
    if hierarchy_edges:
        _apply_hierarchy(cnf, var_of, J, list(hierarchy_edges), hierarchy_transitive)

    # 5) lex chain on requested rows (default all rows)
    if lexi and K > 1:
        rows = list(range(J)) if lexi_row_order is None else list(lexi_row_order)
        if any((r < 0 or r >= J) for r in rows):
            raise ValueError("lexi_row_order contains out-of-range row indices.")
        add_lex_chain_columns(
            cnf, vpool, var_of,
            J=J, K=K,
            rows=rows,
            ascending=bool(lexi_ascending),
            strict=bool(lexi_strict),
        )

    meta = {
        "var_of": var_of,
        "shape": (J, K),
        "n_identity_rows": len(id_cols_list),
        "vpool": vpool,
    }
    return cnf, vpool, meta


# -------------------------
# core samplers
# -------------------------

def _sample_assumption_sat_pool(
    J: int,
    K: int,
    cnf: CNF,
    var_of,
    x_vars: List[int],
    *,
    solver_name: str,
    n_cand: int,
    seed: int,
    assumption_frac: float,
    tries_per_start: int,
) -> List[np.ndarray]:
    """
    Use a single incremental SAT solver; for each candidate:
      - try random assumptions on a random subset of free x-vars
      - solve, read model, block exact x-assignment
    """
    rng = np.random.default_rng(seed)

    xset = set(x_vars)
    fixed = _extract_fixed_xvars(cnf, xset)
    free_vars = [v for v in x_vars if v not in fixed]
    if not free_vars:
        free_vars = x_vars[:]  # fallback

    m = max(1, int(round(assumption_frac * len(free_vars))))
    m = min(m, len(free_vars))

    Qs: List[np.ndarray] = []
    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as S:
        for _ in range(n_cand):
            model_set = None

            for __ in range(tries_per_start):
                subset = rng.choice(free_vars, size=m, replace=False)
                assumps = [int(v) * (1 if rng.random() < 0.5 else -1) for v in subset]
                if S.solve(assumptions=assumps):
                    model_set = set(S.get_model())
                    break

            if model_set is None:
                if not S.solve():
                    break
                model_set = set(S.get_model())

            Q = _model_to_Q(model_set, var_of, J, K)
            Qs.append(Q)

            # block exact assignment of decision vars
            block = [(-v if v in model_set else v) for v in x_vars]
            S.add_clause(block)

    return Qs


def _sample_maxsat_pool(
    J: int,
    K: int,
    cnf: CNF,
    var_of,
    x_vars: List[int],
    *,
    n_cand: int,
    seed: int,
    prefer_ones: bool,
    weight_low: int,
    weight_high: int,
    block_exact: bool = True,
) -> List[np.ndarray]:
    """
    RC2 MaxSAT pool:
      - hard: cnf clauses (+ any accumulated blocking clauses)
      - soft: prefer x=0 via (-x) if prefer_ones=False; else prefer x=1 via (x)
      - random weights for diversity
    """
    rng = np.random.default_rng(seed)
    Qs: List[np.ndarray] = []
    hard_extra: List[List[int]] = []

    for _ in range(n_cand):
        wcnf = WCNF()
        for cl in cnf.clauses:
            wcnf.append(cl)
        for cl in hard_extra:
            wcnf.append(cl)

        wts = rng.integers(weight_low, weight_high + 1, size=len(x_vars))
        for v, w in zip(x_vars, wts):
            wcnf.append([v] if prefer_ones else [-v], weight=int(w))

            
        with RC2(wcnf) as rc2:
            sol = rc2.compute()
            if sol is None:
                # Hard part is UNSAT (often means we've blocked all feasible x-assignments)
                break
            model = set(sol)

        Q = _model_to_Q(model, var_of, J, K)
        Qs.append(Q)

        if block_exact:
            block = [(-v if v in model else v) for v in x_vars]
            hard_extra.append(block)

    return Qs


def sample_Q_init_with_sat(
    J: int,
    K: int,
    q_sat_cfg: QSatConfig,
    *,
    seed: int = 0,
    randomize_clause_literals: bool = True,
) -> np.ndarray:
    """
    One feasible sample via SAT (no assumptions, no MaxSAT).
    """

    cfg_eff = copy.copy(q_sat_cfg)

    cnf, vpool, meta = build_Q_sat_from_cfg(
        J, K,
        include_identity=cfg_eff.include_identity,
        identity_allowed_cols=cfg_eff.identity_allowed_cols,
        include_distinctness=cfg_eff.include_distinctness,
        distinctness_use_indicators=cfg_eff.distinctness_use_indicators,
        col_lb=cfg_eff.col_lb,
        col_up=cfg_eff.col_up,
        row_lb=cfg_eff.row_lb,
        row_up=cfg_eff.row_up,
        lexi=cfg_eff.lexi,
        lexi_ascending=cfg_eff.lexi_ascending,
        lexi_strict=cfg_eff.lexi_strict,
        lexi_row_order=cfg_eff.lexi_row_order,
        hierarchy_edges=cfg_eff.hierarchy_edges,
        hierarchy_transitive=cfg_eff.hierarchy_transitive,
    )

    if randomize_clause_literals:
        import random
        rr = random.Random(seed)
        for cl in cnf.clauses:
            rr.shuffle(cl)

    with Solver(name=cfg_eff.solver_name, bootstrap_with=cnf.clauses) as S:
        if not S.solve():
            raise RuntimeError("SAT initializer: constraints are UNSAT; cannot build Q_init.")
        model = set(S.get_model())

    return _model_to_Q(model, meta["var_of"], J, K)


# -------------------------
# mixed sampler (public)
# -------------------------

def sample_mixed_Q_inits(
    J: int,
    K: int,
    q_sat_cfg: QSatConfig,
    *,
    n_starts: int,
    seed: int = 0,
) -> List[np.ndarray]:
    """
    Produce up to n_starts initial Q matrices using:
      - assumptions SAT (diverse, not-necessarily-sparse)
      - sparse MaxSAT (encourage zeros)
      - optional dense MaxSAT (encourage ones)

    Then:
      - deduplicate
      - enforce minimum Hamming distance between accepted starts

    Returns as many as it can find; guarantees at least one feasible Q
    unless the constraints are UNSAT.
    """
    if n_starts <= 0:
        return []

    cfg_eff = copy.copy(q_sat_cfg)
    cnf, vpool, meta = build_Q_sat_from_cfg(
        J, K,
        include_identity=cfg_eff.include_identity,
        identity_allowed_cols=cfg_eff.identity_allowed_cols,
        include_distinctness=cfg_eff.include_distinctness,
        distinctness_use_indicators=cfg_eff.distinctness_use_indicators,
        col_lb=cfg_eff.col_lb,
        col_up=cfg_eff.col_up,
        row_lb=cfg_eff.row_lb,
        row_up=cfg_eff.row_up,
        lexi=cfg_eff.lexi,
        lexi_ascending=cfg_eff.lexi_ascending,
        lexi_strict=cfg_eff.lexi_strict,
        lexi_row_order=cfg_eff.lexi_row_order,
        hierarchy_edges=cfg_eff.hierarchy_edges,
        hierarchy_transitive=cfg_eff.hierarchy_transitive,
    )

    var_of = meta["var_of"]
    x_vars = [var_of(r, c) for r in range(J) for c in range(K)]

    min_hamming = int(math.ceil(cfg_eff.min_hamming_frac * (J * K)))
    attempt_mult = max(1, int(math.ceil(cfg_eff.max_attempt_factor)))

    n_assump, n_sparse, n_dense = _alloc_counts(
        n_starts,
        cfg_eff.frac_assumption_sat,
        cfg_eff.frac_sparse_maxsat,
        cfg_eff.frac_dense_maxsat,
    )

    # oversample candidates, then filter
    cand_assump = max(n_assump, 1) * attempt_mult
    cand_sparse = max(n_sparse, 0) * attempt_mult
    cand_dense  = max(n_dense, 0) * attempt_mult

    base_seed = int(seed) + int(cfg_eff.seed_offset)

    candidates: List[np.ndarray] = []

    # 1) assumptions SAT pool
    if n_assump > 0:
        candidates.extend(_sample_assumption_sat_pool(
            J, K, cnf, var_of, x_vars,
            solver_name=cfg_eff.solver_name,
            n_cand=cand_assump,
            seed=base_seed + 101,
            assumption_frac=float(cfg_eff.assumption_frac),
            tries_per_start=int(cfg_eff.tries_per_start),
        ))

    # 2) sparse MaxSAT pool
    if n_sparse > 0:
        candidates.extend(_sample_maxsat_pool(
            J, K, cnf, var_of, x_vars,
            n_cand=cand_sparse,
            seed=base_seed + 202,
            prefer_ones=False,
            weight_low=int(cfg_eff.sparse_weight_low),
            weight_high=int(cfg_eff.sparse_weight_high),
            block_exact=True,
        ))

    # 3) optional dense MaxSAT pool
    if n_dense > 0:
        candidates.extend(_sample_maxsat_pool(
            J, K, cnf, var_of, x_vars,
            n_cand=cand_dense,
            seed=base_seed + 303,
            prefer_ones=True,
            weight_low=int(cfg_eff.dense_weight_low),
            weight_high=int(cfg_eff.dense_weight_high),
            block_exact=True,
        ))

    # 4) filter: dedup + min-Hamming
    accepted = _dedup_and_min_hamming(
        candidates,
        n_want=n_starts,
        min_hamming=min_hamming,
        dedup=bool(cfg_eff.dedup),
    )

    # 5) if still short, keep drawing plain SAT solutions (no assumptions) with different seeds
    #    (still respecting dedup + min-Hamming)
    if len(accepted) < n_starts:
        extra_cands: List[np.ndarray] = []
        # bounded tries
        max_extra = max(5, 10 * (n_starts - len(accepted)))
        for t in range(max_extra):
            Q = sample_Q_init_with_sat(J, K, cfg_eff, seed=base_seed + 1000 + t, randomize_clause_literals=True)
            extra_cands.append(Q)
        accepted = _dedup_and_min_hamming(
            accepted + extra_cands,
            n_want=n_starts,
            min_hamming=min_hamming,
            dedup=bool(cfg_eff.dedup),
        )

    # guarantee at least one if SAT is feasible
    if not accepted:
        accepted = [sample_Q_init_with_sat(J, K, cfg_eff, seed=base_seed + 9999)]

    return accepted


# -------------------------
# Backward-compatible name used by em.py
# -------------------------

def sample_random_Q_inits_with_sat(
    J: int,
    K: int,
    q_sat_cfg: QSatConfig,
    *,
    n_starts: int,
    seed: int = 0,
    **kwargs,
) -> List[np.ndarray]:
    """
    Backward-compatible wrapper.
    Ignores legacy kwargs like pool_max.
    """
    return sample_mixed_Q_inits(J, K, q_sat_cfg, n_starts=n_starts, seed=seed)
