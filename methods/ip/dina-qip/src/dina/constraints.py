# ============================ src/qem_dina/constraints.py =====================

from typing import Optional, Sequence, Dict, Tuple, List
from gurobipy import Model, GRB, quicksum


def _validate_vec(vec, length, name):
    if vec is None: return [None]*length
    assert len(vec) == length, f"{name} must have length {length}"
    return list(vec)


def add_completeness(m: Model, q, J, K, *, allowed_cols: Optional[Sequence[bool]] = None, name_prefix="id"):
    y = m.addVars(J, K, vtype=GRB.BINARY, name=f"{name_prefix}_y")
    if allowed_cols is None: allowed_cols = [True]*K
    for k, ok in enumerate(allowed_cols):
        m.addConstr(quicksum(y[j, k] for j in range(J)) == (1 if ok else 0), name=f"{name_prefix}_pick_col{k}")
    for j in range(J):
        m.addConstr(quicksum(y[j, k] for k in range(K)) <= 1, name=f"{name_prefix}_use_row{j}")
    for j in range(J):
        for k in range(K):
            m.addConstr(q[j, k] >= y[j, k], name=f"{name_prefix}_diag_j{j}_k{k}")
            for ell in range(K):
                if ell == k: continue
                m.addConstr(q[j, ell] <= 1 - y[j, k], name=f"{name_prefix}_off_j{j}_k{k}_l{ell}")
    row_is_id = {j: quicksum(y[j, k] for k in range(K)) for j in range(J)}
    return y, row_is_id


def add_distinctness(m: Model, q, row_is_id, J, K, *, name_prefix="dist", use_indicators=False):
    if K <= 1: return {}
    r = m.addVars(J, K, K, vtype=GRB.BINARY, name=f"{name_prefix}_r")
    for k in range(K-1):
        for ell in range(k+1, K):
            m.addConstr(quicksum(r[j, k, ell] for j in range(J)) >= 1, name=f"{name_prefix}_some_k{k}_l{ell}")
            for j in range(J):
                m.addConstr(r[j, k, ell] <= 1 - row_is_id[j], name=f"{name_prefix}_nonid_j{j}_k{k}_l{ell}")
                if use_indicators:
                    m.addGenConstrIndicator(r[j, k, ell], True, q[j, k] + q[j, ell] == 1, name=f"{name_prefix}_xor_ind_j{j}_k{k}_l{ell}")
                else:
                    m.addConstr(r[j, k, ell] <= q[j, k] + q[j, ell], name=f"{name_prefix}_xor_hi_j{j}_k{k}_l{ell}")
                    m.addConstr(r[j, k, ell] <= 2 - (q[j, k] + q[j, ell]), name=f"{name_prefix}_xor_lo_j{j}_k{k}_l{ell}")
    return r


def add_col_bounds(m: Model, q, J: int, K: int, lb: Optional[Sequence[Optional[int]]] = None, ub: Optional[Sequence[Optional[int]]] = None, name_prefix: str = "col"):
    lb = _validate_vec(lb, K, "col lb"); ub = _validate_vec(ub, K, "col ub"); cons = {}
    for k in range(K):
        expr = quicksum(q[j, k] for j in range(J))
        if lb[k] is not None: cons[(k, "lb")] = m.addConstr(expr >= int(lb[k]), name=f"{name_prefix}_lb_k{k}")
        if ub[k] is not None: cons[(k, "ub")] = m.addConstr(expr <= int(ub[k]), name=f"{name_prefix}_ub_k{k}")
    return cons


def add_row_bounds(m: Model, q, J: int, K: int, lb: Optional[Sequence[Optional[int]]] = None, ub: Optional[Sequence[Optional[int]]] = None, name_prefix: str = "row"):
    lb = _validate_vec(lb, J, "row lb"); ub = _validate_vec(ub, J, "row ub"); cons = {}
    for j in range(J):
        expr = quicksum(q[j, k] for k in range(K))
        if lb[j] is not None: cons[(j, "lb")] = m.addConstr(expr >= int(lb[j]), name=f"{name_prefix}_lb_j{j}")
        if ub[j] is not None: cons[(j, "ub")] = m.addConstr(expr <= int(ub[j]), name=f"{name_prefix}_ub_j{j}")
    return cons



def add_lexi(
    m: Model,
    q,
    J,
    K,
    *,
    ascending=False,
    strict=False,
    row_order=None,
    name_prefix="lex",
):
    """
    Enforce exact lexicographic order on adjacent columns of q.

    By default (ascending=False), enforce
        q[:,0] >=_lex q[:,1] >=_lex ... >=_lex q[:,K-1]
    with respect to the row order given by `row_order`, whose first entry is
    the most significant row in the lex comparison.

    If ascending=True, enforce
        q[:,0] <=_lex q[:,1] <=_lex ... <=_lex q[:,K-1].

    If strict=True, enforce strict lex order between each adjacent pair, i.e.
    equality of two adjacent columns is forbidden.

    Parameters
    ----------
    m : gurobipy.Model
    q : indexable collection of binary vars, accessed as q[j, k]
    J : int
        Number of rows.
    K : int
        Number of columns.
    ascending : bool, default False
    strict : bool, default False
    row_order : list[int] or None
        Order of rows used in lex comparison; first element is most significant.
        Defaults to [0, 1, ..., J-1].
    name_prefix : str, default "lex"

    Returns
    -------
    evars : dict
        Dictionary mapping k -> list of vars e[k][r], r=0..J, where
        e[k][r] = 1 iff columns k and k+1 are equal on the first r rows
        of `row_order`.
    """
    if K <= 1:
        return {}

    if row_order is None:
        row_order = list(range(J))
    if len(row_order) != J:
        raise ValueError("row_order must have length J")
    if sorted(row_order) != list(range(J)):
        raise ValueError("row_order must be a permutation of 0..J-1")

    evars = {}

    for k in range(K - 1):
        # e[r] = 1 iff columns k and k+1 are equal on the first r rows of row_order
        e = [
            m.addVar(lb=1.0, ub=1.0, name=f"{name_prefix}_eq_k{k}_r0")
        ]
        e += [
            m.addVar(lb=0.0, ub=1.0, name=f"{name_prefix}_eq_k{k}_r{r}")
            for r in range(1, J + 1)
        ]
        evars[k] = e

        for r in range(1, J + 1):
            j = row_order[r - 1]
            a = q[j, k]
            b = q[j, k + 1]

            # Main lex constraint at the first differing row.
            # If equal so far (e[r-1]=1), then:
            #   descending: forbid (a,b) = (0,1), i.e. require a >= b
            #   ascending : forbid (a,b) = (1,0), i.e. require a <= b
            if ascending:
                m.addConstr(
                    a - b <= 1 - e[r - 1],
                    name=f"{name_prefix}_main_asc_k{k}_r{r}",
                )
            else:
                m.addConstr(
                    b - a <= 1 - e[r - 1],
                    name=f"{name_prefix}_main_desc_k{k}_r{r}",
                )

            # Propagate prefix equality:
            # e[r] = e[r-1] AND (a == b)
            m.addConstr(
                e[r] <= e[r - 1],
                name=f"{name_prefix}_u1_k{k}_r{r}",
            )
            m.addConstr(
                e[r] <= 1 - a + b,
                name=f"{name_prefix}_u2_k{k}_r{r}",
            )
            m.addConstr(
                e[r] <= 1 + a - b,
                name=f"{name_prefix}_u3_k{k}_r{r}",
            )
            m.addConstr(
                e[r] >= e[r - 1] - a - b,
                name=f"{name_prefix}_l1_k{k}_r{r}",
            )
            m.addConstr(
                e[r] >= e[r - 1] + a + b - 2,
                name=f"{name_prefix}_l2_k{k}_r{r}",
            )

        # Strict lex order means the two columns cannot be identical.
        if strict:
            m.addConstr(
                e[J] == 0,
                name=f"{name_prefix}_strict_k{k}",
            )

    return evars


def _check_dag(K: int, edges: List[Tuple[int,int]]):
    indeg = [0]*K; adj = [[] for _ in range(K)]
    for u,v in edges:
        if not (0<=u<K and 0<=v<K) or u==v: raise ValueError("Invalid hierarchy edge.")
        adj[u].append(v); indeg[v]+=1
    q = [i for i in range(K) if indeg[i]==0]; seen = 0
    while q:
        u = q.pop(); seen += 1
        for w in adj[u]:
            indeg[w]-=1
            if indeg[w]==0: q.append(w)
    if seen != K:
        raise ValueError("Hierarchy contains a cycle; must be a DAG.")


def _closure(K: int, edges: List[Tuple[int,int]]):
    reach = [[False]*K for _ in range(K)]
    for u,v in edges: reach[u][v] = True
    for k in range(K):
        for i in range(K):
            if reach[i][k]:
                for j in range(K): reach[i][j] = reach[i][j] or reach[k][j]
    return [(i,j) for i in range(K) for j in range(K) if reach[i][j]]


def add_hierarchy(m: Model, q, J, K, edges: List[Tuple[int,int]], *, transitive=True, name_prefix="hier"):
    _check_dag(K, edges)
    E = _closure(K, edges) if transitive else edges
    for (u,v) in E:
        for j in range(J):
            m.addConstr(q[j,u] <= q[j,v], name=f"{name_prefix}_j{j}_u{u}_v{v}")
