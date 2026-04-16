# src/qem_dina/SAT_constraints.py
from __future__ import annotations

from typing import Optional, Sequence, List, Tuple
from pysat.formula import CNF, IDPool


# -------------------------
# small utilities
# -------------------------
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


# -------------------------
# small utilities
# -------------------------

def _coerce_optional_int(x: Optional[object], *, name: str) -> Optional[int]:
    """
    Convert x to a *Python* int if possible; keep None.
    Raise if x is not integer-like (e.g., 1.2).
    This prevents PySAT/CardEnc from throwing "integer expected" later.
    """
    if x is None:
        return None

    # bool is int-like but usually unintended; still allow (0/1)
    if isinstance(x, bool):
        return int(x)

    # plain python int
    if isinstance(x, int):
        return x

    # numpy scalar?
    if np is not None:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            xf = float(x)
            if xf.is_integer():
                return int(xf)
            raise ValueError(f"{name} must be an integer; got float-like {x!r}")

    # python float
    if isinstance(x, float):
        if x.is_integer():
            return int(x)
        raise ValueError(f"{name} must be an integer; got float {x!r}")

    # anything else (e.g., np.ndarray, str)
    raise TypeError(f"{name} must be int-like or None; got {type(x)} with value {x!r}")


def _as_bounds_vec(
    x: Optional[Sequence[Optional[object]]],
    L: int,
    default: Optional[int] = None,
    *,
    name: str = "bounds",
) -> List[Optional[int]]:
    """
    Normalize a bounds vector to a Python list of length L with elements in {None} ∪ Z.
    Coerces numpy scalar ints/floats like np.int64 or 2.0 to Python int.
    """
    if x is None:
        return [default] * L
    if len(x) != L:
        raise ValueError(f"{name}: expected length {L}, got {len(x)}")

    out: List[Optional[int]] = []
    for i, xi in enumerate(x):
        out.append(_coerce_optional_int(xi, name=f"{name}[{i}]"))
    return out



def _xor_equiv(cnf: CNF, v_y: int, v_a: int, v_b: int):
    # y <-> (a XOR b)
    cnf.extend([
        [-v_y,  v_a,  v_b],
        [-v_y, -v_a, -v_b],
        [ v_y,  v_a, -v_b],
        [ v_y, -v_a,  v_b],
    ])


# -------------------------
# Lex constraints (carry encoding)
# -------------------------

def add_lex_ge(cnf: CNF, vpool: IDPool, X: Sequence[int], Y: Sequence[int], *, strict: bool, tag=("lex",)):
    """
    Enforce X >=_lex Y (and if strict: X >_lex Y).
    X,Y are lists of literals from most-significant to least-significant.
    Convention: 1 > 0. Lex violation at first differing bit is (X_k=0, Y_k=1).

    Uses carry vars t_k meaning "prefix equal up to k-1" (t_0 = True).
    """
    n = len(X)
    if n != len(Y):
        raise ValueError("add_lex_ge: X and Y must have the same length.")
    if n == 0:
        return

    def imp(a: int, clause: Sequence[int]):
        cnf.append([-a] + list(clause))

    # t_1,...,t_n
    t = [None] + [vpool.id(tag + ("t", k)) for k in range(1, n + 1)]

    # At k=0, tied is true. Forbid wrong direction: (X0 ∨ ¬Y0)
    cnf.append([X[0], -Y[0]])

    # t1 <-> (X0 == Y0)
    # t1 -> equality
    imp(t[1], [-X[0], Y[0]])
    imp(t[1], [-Y[0], X[0]])
    # equality -> t1
    cnf.append([ X[0],  Y[0],  t[1]])
    cnf.append([-X[0], -Y[0],  t[1]])

    # For k=1..n-1:
    # if tied so far (t_k), forbid wrong direction at bit k: (Xk ∨ ¬Yk)
    # and define t_{k+1} <-> (t_k & (Xk==Yk)).
    for k in range(1, n):
        # t_k -> (Xk ∨ ¬Yk)
        imp(t[k], [X[k], -Y[k]])

        # t_{k+1} -> t_k and equality
        imp(t[k + 1], [t[k]])
        imp(t[k + 1], [-X[k], Y[k]])
        imp(t[k + 1], [-Y[k], X[k]])

        # (t_k & Xk & Yk) -> t_{k+1}
        cnf.append([-t[k], -X[k], -Y[k], t[k + 1]])
        # (t_k & ¬Xk & ¬Yk) -> t_{k+1}
        cnf.append([-t[k],  X[k],  Y[k], t[k + 1]])

    if strict:
        # forbid full equality: OR_k (Xk XOR Yk)
        diff_lits: List[int] = []
        for k in range(n):
            d = vpool.id(tag + ("d", k))
            _xor_equiv(cnf, d, X[k], Y[k])
            diff_lits.append(d)
        cnf.append(diff_lits)


def add_lex_chain_columns(
    cnf: CNF,
    vpool: IDPool,
    var_of,
    *,
    J: int,
    K: int,
    rows: Optional[Sequence[int]] = None,
    ascending: bool = False,
    strict: bool = False,
):
    """
    Enforce a lex chain across adjacent columns.
      - ascending=True:  col0 <=_lex col1 <=_lex ... <=_lex col(K-1)
      - ascending=False: col0 >=_lex col1 >=_lex ... >=_lex col(K-1)

    rows controls which rows participate; if None, uses all rows 0..J-1.
    """
    if K <= 1:
        return
    if rows is None:
        rows = list(range(J))
    else:
        rows = list(rows)

    for c in range(K - 1):
        col_c  = [var_of(r, c)     for r in rows]
        col_c1 = [var_of(r, c + 1) for r in rows]

        if ascending:
            # col_c <=_lex col_c1  <=>  col_c1 >=_lex col_c
            add_lex_ge(cnf, vpool, col_c1, col_c, strict=strict, tag=("lex", "col", c))
        else:
            # col_c >=_lex col_c1
            add_lex_ge(cnf, vpool, col_c, col_c1, strict=strict, tag=("lex", "col", c))


# -------------------------
# hierarchy
# -------------------------

def _apply_hierarchy(cnf: CNF, var_of, J: int, edges: List[Tuple[int, int]], transitive: bool):
    """
    Each edge is (parent, child) meaning parent prerequisite for child:
    enforce q[j,child] -> q[j,parent] for all rows j.
    """
    if not edges:
        return

    K = max([max(u, v) for (u, v) in edges] + [0]) + 1
    reach = [[False] * K for _ in range(K)]
    for u, v in edges:
        reach[u][v] = True

    if transitive:
        for m in range(K):
            for i in range(K):
                if reach[i][m]:
                    rim = reach[i]
                    rmm = reach[m]
                    for j in range(K):
                        rim[j] = rim[j] or rmm[j]

    for u in range(K):
        for v in range(K):
            if reach[u][v]:
                for j in range(J):
                    cnf.append([-var_of(j, v), var_of(j, u)])
