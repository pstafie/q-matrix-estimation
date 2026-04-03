# =============================== src/qem_dina/q_mip.py ========================

import math
import numpy as np
from typing import Optional, Sequence, Dict, Tuple, Any
from gurobipy import Model, GRB, quicksum

from .constraints import (
    add_completeness, add_distinctness, add_col_bounds, add_row_bounds,
    add_lexi, add_hierarchy
)


def mstep_update_Q(
    R: np.ndarray, s: np.ndarray, g: np.ndarray, nu: np.ndarray, tau: np.ndarray, Q_start: np.ndarray,
    **q_options: Any
):
    N, J = R.shape
    P = tau.shape[1]
    K = int(round(math.log2(P)))
    assert 1<<K == P
    assert s.shape == (J,) and g.shape == (J,)

    s_clip = np.clip(s, q_options.get('clip_eps', 1e-12), 1- q_options.get('clip_eps', 1e-12))
    g_clip = np.clip(g, q_options.get('clip_eps', 1e-12), 1- q_options.get('clip_eps', 1e-12))
    beta = R * np.log((1 - s_clip)[None, :] / g_clip[None, :]) + (1 - R) * np.log(s_clip[None, :] / (1 - g_clip)[None, :])
    w = beta.T @ tau    # (J,P)

    m = Model("Mstep_Q")
    m.Params.OutputFlag = 1 if q_options.get('log_to_console', True) else 0
    if q_options.get('time_limit') is not None:
        m.Params.TimeLimit = q_options['time_limit']
    if q_options.get('mip_gap') is not None:
        m.Params.MIPGap = q_options['mip_gap']

    q = m.addVars(J, K, vtype=GRB.BINARY, name="q")
    # xi = m.addVars(J, P, vtype=GRB.BINARY, name="xi")
    xi = m.addVars(J, P, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="xi")
    
    zeros_by_a = [[k for k in range(K) if ((a >> k) & 1) == 0] for a in range(P)]
    for a in range(P):
        zeros = zeros_by_a[a]
        for j in range(J):
            for k in zeros:
                m.addConstr(xi[j, a] <= 1 - q[j, k], name=f"xi_up_j{j}_a{a}_k{k}")
            if zeros:
                m.addConstr(xi[j, a] >= 1 - quicksum(q[j, k] for k in zeros), name=f"xi_low_j{j}_a{a}")
            else:
                m.addConstr(xi[j, a] == 1, name=f"xi_allones_j{j}_a{a}")

    m.setObjective(quicksum(w[j, a] * xi[j, a] for j in range(J) for a in range(P)), GRB.MAXIMIZE)

    row_is_id = {j: 0 for j in range(J)}
    y = None
    if q_options.get('include_identity', False):
        y, row_is_id = add_completeness(m, q, J, K, allowed_cols=q_options.get('identity_allowed_cols'), name_prefix="id")

    if q_options.get('include_distinctness', False):
        _ = add_distinctness(m, q, row_is_id, J, K, name_prefix="dist", use_indicators=q_options.get('distinctness_use_indicators', False))

    if (q_options.get('col_lb') is not None) or (q_options.get('col_up') is not None):
        add_col_bounds(m, q, J, K, lb=q_options.get('col_lb'), ub=q_options.get('col_up'), name_prefix="col")

    if (q_options.get('row_lb') is not None) or (q_options.get('row_up') is not None):
        add_row_bounds(m, q, J, K, lb=q_options.get('row_lb'), ub=q_options.get('row_up'), name_prefix="row")

    if q_options.get('lexi', False):
        _ = add_lexi(m, q, J, K, ascending=q_options.get('lexi_ascending', False), strict=q_options.get('lexi_strict', False), row_order=q_options.get('lexi_row_order'), name_prefix="lex")

    if q_options.get('hierarchy_edges'):
        add_hierarchy(m, q, J, K, q_options['hierarchy_edges'], transitive=q_options.get('hierarchy_transitive', True), name_prefix="hier")

    if Q_start is not None:
        Q_start = (np.asarray(Q_start) > 0).astype(int)
        assert Q_start.shape == (J, K)
        for j in range(J):
            for k in range(K):
                q[j, k].Start = int(Q_start[j, k])
        for a in range(P):
            zeros = [k for k in range(K) if ((a >> k) & 1) == 0]
            for j in range(J):
                violates = any(Q_start[j, k] == 1 for k in zeros)
                xi[j, a].Start = 0 if violates else 1

    m.optimize()

    Q_sol = np.zeros((J, K), dtype=int)
    if m.SolCount > 0:
        for j in range(J):
            for k in range(K):
                Q_sol[j, k] = int(round(q[j, k].X))

    return {"Q": Q_sol, "obj": m.ObjVal if m.SolCount > 0 else None, "status": m.Status, "model": m, "vars": {"q": q, "xi": xi, "y": y}}