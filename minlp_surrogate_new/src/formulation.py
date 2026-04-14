"""
formulation.py
==============
All AC Unit Commitment optimisation models and training-data generation.

Mirrors formulation.jl (all functions) and generate_training_data() from
test.ipynb.

Solver mapping
--------------
Julia used JuMP with the following solvers:
    Rectangular formulations  ->  Gurobi   (MIQCQP / QCP)
    Polar formulations        ->  Ipopt    (NL, via @NLconstraint)

Python mirrors this exactly:
    Rectangular formulations  ->  gurobipy  (Gurobi Python API)
    Polar formulations        ->  Pyomo + Ipopt

Note on polar binary variables
-------------------------------
Ipopt is a continuous NLP solver and cannot handle binary u directly.
The polar models therefore *relax* u to [0, 1] continuous, exactly as the
Julia code effectively did (Gurobi's @NLconstraint support is limited and
the Julia polar formulation was also implicitly solving an NLP relaxation).
For MINLP polar, swap the Ipopt solver for SCIP or Bonmin in pyomo.

Public API  (mirrors Julia)
---------------------------
    ac_uc(data, type="Rectangular")
    mp_ac_uc(data, demand_curve, type="Rectangular")
    build_single_ac_uc_rectangular(data)
    build_single_period_ac_uc_polar(data)
    build_mp_ac_uc_rectangular(data, demand_curve)
    build_mp_ac_uc_polar(data, demand_curve)
    build_convex_ac_uc(data, node_vr, node_vi, node_pd, node_qd)
    get_old_voltages(model, data)          <- re-exported from data_utils
    generate_training_data(data, node_vr, node_vi, n_samples)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from .data_utils import (
    MatpowerData,
    calc_branch_y,
    calc_branch_t,
    get_edges,
    get_old_voltages,   # re-export so callers only need formulation
)


# ===========================================================================
# Public dispatcher functions  (mirror Julia ac_uc / mp_ac_uc)
# ===========================================================================

def ac_uc(data: MatpowerData, formulation: str = "Rectangular"):
    """
    Single-period AC-UC dispatcher.
    formulation = "Rectangular" | "Polar"
    Mirrors: ac_uc(file_path, type) in Julia.
    """
    if formulation == "Rectangular":
        return build_single_ac_uc_rectangular(data)
    else:
        return build_single_period_ac_uc_polar(data)


def mp_ac_uc(
    data: MatpowerData,
    demand_curve: List[float],
    formulation: str = "Rectangular",
):
    """
    Multi-period AC-UC dispatcher.
    Mirrors: mp_ac_uc(file_path, demand_curve, type) in Julia.
    """
    if formulation == "Rectangular":
        return build_mp_ac_uc_rectangular(data, demand_curve)
    else:
        return build_mp_ac_uc_polar(data, demand_curve)


# ===========================================================================
# ── GUROBIPY  (Rectangular formulations + Convex relaxation) ────────────────
# ===========================================================================

# ---------------------------------------------------------------------------
# build_single_ac_uc_rectangular
# ---------------------------------------------------------------------------

def build_single_ac_uc_rectangular(data: MatpowerData) -> gp.Model:
    """
    Single-period non-convex MIQCQP AC-UC in rectangular / W-matrix form.

    Used to obtain warm-start voltages (vr, vi) for the convex solve.
    Mirrors Julia's build_single_ac_uc_rectangular(file_path).

    Solver: Gurobi  (requires NonConvex=2 for the bilinear W-matrix equalities)
    """
    T = [1]
    m = gp.Model("single_ac_uc_rect")
    m.Params.NonConvex = 2   # needed for quadratic equality constraints

    _add_acuc_var_rectangular(m, data, T, convex=False)
    _add_mincost_obj_gurobi(m, data, T, convex=False)
    _add_ref_limits_rectangular(m, data, T)
    _add_gen_limits_gurobi(m, data, T)
    _add_rectangular_branchflow(m, data, T)
    _add_node_bal_rectangular(m, data, T)
    return m


# ---------------------------------------------------------------------------
# build_mp_ac_uc_rectangular
# ---------------------------------------------------------------------------

def build_mp_ac_uc_rectangular(
    data: MatpowerData,
    demand_curve: List[float],
) -> gp.Model:
    """
    Multi-period non-convex MIQCQP AC-UC in rectangular / W-matrix form.
    demand_curve[t] is the demand scale factor for period t.
    Mirrors Julia's build_mp_ac_uc_rectangular(file_path, demand_curve).
    """
    T = list(range(1, len(demand_curve) + 1))
    m = gp.Model("mp_ac_uc_rect")
    m.Params.NonConvex = 2

    _add_acuc_var_rectangular(m, data, T, convex=False)
    _add_mincost_obj_gurobi(m, data, T, convex=False)
    _add_ref_limits_rectangular(m, data, T)
    _add_gen_limits_gurobi(m, data, T)
    _add_rectangular_branchflow(m, data, T)
    _add_node_bal_rectangular(m, data, T, demand_curve=demand_curve)
    return m


# ---------------------------------------------------------------------------
# build_convex_ac_uc
# ---------------------------------------------------------------------------

def build_convex_ac_uc(
    data: MatpowerData,
    node_vr: Dict[str, float],
    node_vi: Dict[str, float],
    node_pd: Optional[Dict[str, float]] = None,
    node_qd: Optional[Dict[str, float]] = None,
    conn: Optional[Dict] = None,
    penalty_weight: float = 100000.0  # Added type hint for safety
) -> gp.Model:
    """
    Constante-Flores & Li (2026) convex QCQP relaxation of AC-UC.

    Uses warm-start voltages (node_vr, node_vi) to linearise the W-matrix
    definitions (constraints 4b–4j in the paper).  The result is a convex
    QCP with binary u that Gurobi can solve without NonConvex=2.

    node_pd / node_qd  – per-bus active / reactive load (pu) for this sample.
                         If None, the case-file baseline loads are used.
    conn               – pre-computed connectivity dict (pass in to avoid
                         recomputing when calling inside generate_training_data).

    Constraint references are stored on the returned model:
        m._pbal[bus_id, 1].RHS  – active balance RHS  (= pd per bus)
        m._qbal[bus_id, 1].RHS  – reactive balance RHS (= qd per bus)

    Mirrors Julia's build_convex_ac_uc(file_path, node_vr, node_vi, node_pd, node_qd).
    """
    T = [1]
    m = gp.Model("convex_ac_uc")
    m.setParam('OutputFlag', 0)      # suppress Gurobi banner by default
    m.setParam('BarHomogeneous', 1)  # robust barrier for ill-conditioned QCQP
    m.setParam('NumericFocus', 1)    # extra numerical care
    m.setParam('QCPDual', 0)         # skip QCP dual (faster, avoids numerical issues)

    _add_acuc_var_rectangular(m, data, T, convex=True)
    _add_convex_constraints(m, data, T, node_vr, node_vi)
    # FIX: Pass the penalty_weight down to the objective builder
    _add_mincost_obj_gurobi(m, data, T, convex=True, penalty_weight=penalty_weight)
    _add_ref_limits_rectangular(m, data, T)
    _add_gen_limits_gurobi(m, data, T)
    _add_rectangular_branchflow(m, data, T)
    param_loads = {'pd': node_pd, 'qd': node_qd} if (node_pd is not None) else None
    _add_node_bal_rectangular(m, data, T, param_loads=param_loads, conn=conn)
    return m


# ---------------------------------------------------------------------------
# _add_acuc_var_rectangular
# ---------------------------------------------------------------------------

def _add_acuc_var_rectangular(
    m: gp.Model,
    data: MatpowerData,
    T: List[int],
    convex: bool = False,
) -> None:
    """
    Add rectangular / W-matrix variables to a gurobipy model.

    Variables stored as attributes on m (underscore prefix):
        m._vr, m._vi        – real / imaginary voltages   keyed (bus_str, t)
        m._c_ii             – |V|^2 diagonals              keyed (bus_str, t)
        m._c_ij, m._s_ij    – off-diagonals                keyed (f_int, t_int, t)
        m._u                – binary commitment            keyed (gen_str, t)
        m._pg, m._qg        – active/reactive generation  keyed (gen_str, t)
        m._p_fr, m._q_fr    – from-side branch flows      keyed (branch_str, t)
        m._p_to, m._q_to    – to-side branch flows        keyed (branch_str, t)

    convex=False  adds the nonlinear W-matrix equality constraints (1l,1m,1n).
    convex=True   skips them; _add_convex_constraints() adds the relaxed ones.

    Mirrors Julia's _add_acuc_var_rectangular!() in formulation.jl.
    """
    buses    = data.buses
    gens     = data.gens
    branches = data.branches
    edges    = get_edges(data)          # list of (f_int, t_int)

    bus_ids    = list(buses.keys())
    gen_ids    = list(gens.keys())
    branch_ids = list(branches.keys())

    # ---- voltage variables (unbounded real / imaginary) --------------------
    m._vr = m.addVars(bus_ids, T, lb=-GRB.INFINITY, name="vr")
    m._vi = m.addVars(bus_ids, T, lb=-GRB.INFINITY, name="vi")

    # ---- W-matrix diagonal  c_ii ∈ [vmin^2, vmax^2]  (constraint 1i) -----
    m._c_ii = m.addVars(
        bus_ids, T,
        lb={(i, t): buses[i]['vmin'] ** 2 for i in bus_ids for t in T},
        ub={(i, t): buses[i]['vmax'] ** 2 for i in bus_ids for t in T},
        name="c_ii",
    )

    # ---- W-matrix off-diagonals  c_ij, s_ij  (unbounded) ------------------
    edge_t_keys = [(fi, ti, t) for (fi, ti) in edges for t in T]
    m._c_ij = m.addVars(edge_t_keys, lb=-GRB.INFINITY, name="c_ij")
    m._s_ij = m.addVars(edge_t_keys, lb=-GRB.INFINITY, name="s_ij")

    if not convex:
        # Non-convex equality constraints defining the W matrix (1l, 1m, 1n)
        for i in bus_ids:
            for t in T:
                vr = m._vr[i, t]
                vi = m._vi[i, t]
                m.addQConstr(
                    m._c_ii[i, t] == vr * vr + vi * vi,
                    name=f"wii_{i}_{t}",
                )
        for (fi, ti) in edges:
            i = str(fi)
            j = str(ti)
            for t in T:
                vr_i = m._vr[i, t];  vr_j = m._vr[j, t]
                vi_i = m._vi[i, t];  vi_j = m._vi[j, t]
                m.addQConstr(
                    m._c_ij[fi, ti, t] == vr_i * vr_j + vi_i * vi_j,
                    name=f"wij_{fi}_{ti}_{t}",
                )
                m.addQConstr(
                    m._s_ij[fi, ti, t] == vr_i * vi_j - vr_j * vi_i,
                    name=f"sij_{fi}_{ti}_{t}",
                )

    # ---- generation variables ----------------------------------------------
    m._u  = m.addVars(gen_ids, T, vtype=GRB.BINARY,    name="u")
    m._pg = m.addVars(gen_ids, T, lb=-GRB.INFINITY,    name="pg")
    m._qg = m.addVars(gen_ids, T, lb=-GRB.INFINITY,    name="qg")

    # ---- branch flow variables ---------------------------------------------
    m._p_fr = m.addVars(branch_ids, T, lb=-GRB.INFINITY, name="p_fr")
    m._q_fr = m.addVars(branch_ids, T, lb=-GRB.INFINITY, name="q_fr")
    m._p_to = m.addVars(branch_ids, T, lb=-GRB.INFINITY, name="p_to")
    m._q_to = m.addVars(branch_ids, T, lb=-GRB.INFINITY, name="q_to")

    m.update()


# ---------------------------------------------------------------------------
# _add_convex_constraints
# ---------------------------------------------------------------------------

def _add_convex_constraints(
    m: gp.Model,
    data: MatpowerData,
    T: List[int],
    node_vr: Dict[str, float],
    node_vi: Dict[str, float],
) -> None:
    """
    Add the Constante-Flores & Li (2026) convex W-matrix approximation
    constraints (4b – 4j) to a gurobipy model.

    Slack variables:
        m._xi_c  [bus, t]  >= 0    (4h)
        m._xij_c [fi,tj,t] >= 0    (4i)
        m._xij_s [fi,tj,t] >= 0    (4j)

    Mirrors Julia's _add_convex_constraints!() in formulation.jl.
    """
    buses  = data.buses
    edges  = get_edges(data)

    bus_ids     = list(buses.keys())
    edge_t_keys = [(fi, ti, t) for (fi, ti) in edges for t in T]

    # Slack variables (non-negative)
    m._xi_c  = m.addVars(bus_ids,     T, lb=0.0, name="xi_c")
    m._xij_c = m.addVars(edge_t_keys, lb=0.0,    name="xij_c")
    m._xij_s = m.addVars(edge_t_keys, lb=0.0,    name="xij_s")
    m.update()

    # ---- 4b & 4c: diagonal W entries ---------------------------------------
    for i in bus_ids:
        vr0 = node_vr[i]
        vi0 = node_vi[i]
        for t in T:
            vr = m._vr[i, t]
            vi = m._vi[i, t]
            c  = m._c_ii[i, t]

            # 4b: c_ii >= vr^2 + vi^2  (convex, SOC-representable)
            m.addQConstr(
                c >= vr * vr + vi * vi,
                name=f"4b_{i}_{t}",
            )
            # 4c: c_ii <= 2*(vr0*vr + vi0*vi) - (vr0^2 + vi0^2) + xi_c
            #     (linear in vr, vi since vr0, vi0 are fixed parameters)
            m.addConstr(
                c <= 2.0 * (vr0 * vr + vi0 * vi)
                     - (vr0 ** 2 + vi0 ** 2)
                     + m._xi_c[i, t],
                name=f"4c_{i}_{t}",
            )

    # ---- 4d – 4f: off-diagonal W entries -----------------------------------
    for (fi, ti) in edges:
        i  = str(fi)
        j  = str(ti)
        vr0_i = node_vr[i];  vr0_j = node_vr[j]
        vi0_i = node_vi[i];  vi0_j = node_vi[j]
        for t in T:
            vr_i  = m._vr[i, t];  vr_j  = m._vr[j, t]
            vi_i  = m._vi[i, t];  vi_j  = m._vi[j, t]
            c_ij  = m._c_ij[fi, ti, t]
            s_ij  = m._s_ij[fi, ti, t]
            xij_c = m._xij_c[fi, ti, t]
            xij_s = m._xij_s[fi, ti, t]

            # ---- 4d --------------------------------------------------------
            # (vr_i+vr_j)^2 + (vi_i+vi_j)^2 + 4*c_ij
            #   <= xij_c + 2*(vr_i-vr_j)*(vr0_i-vr0_j)
            #            + 2*(vi_i-vi_j)*(vi0_i-vi0_j)
            #            - ((vr0_i-vr0_j)^2 + (vi0_i-vi0_j)^2)
            rhs_4d = (
                xij_c
                + 2.0 * (vr_i - vr_j) * (vr0_i - vr0_j)
                + 2.0 * (vi_i - vi_j) * (vi0_i - vi0_j)
                - ((vr0_i - vr0_j) ** 2 + (vi0_i - vi0_j) ** 2)
            )
            m.addQConstr(
                (vr_i + vr_j) * (vr_i + vr_j)
                + (vi_i + vi_j) * (vi_i + vi_j)
                - 4.0 * c_ij <= rhs_4d,
                name=f"4d_{fi}_{ti}_{t}",
            )

            # ---- 4e --------------------------------------------------------
            # (vr_i-vr_j)^2 + (vi_i-vi_j)^2 + 4*c_ij
            #   <= xij_c + 2*(vr_i+vr_j)*(vr0_i+vr0_j)
            #            + 2*(vi_i+vi_j)*(vi0_i+vi0_j)
            #            - ((vr0_i+vr0_j)^2 + (vi0_i+vi0_j)^2)
            rhs_4e = (
                xij_c
                + 2.0 * (vr_i + vr_j) * (vr0_i + vr0_j)
                + 2.0 * (vi_i + vi_j) * (vi0_i + vi0_j)
                - ((vr0_i + vr0_j) ** 2 + (vi0_i + vi0_j) ** 2)
            )
            m.addQConstr(
                (vr_i - vr_j) * (vr_i - vr_j)
                + (vi_i - vi_j) * (vi_i - vi_j)
                + 4.0 * c_ij <= rhs_4e,
                name=f"4e_{fi}_{ti}_{t}",
            )

            # ---- 4f (first inequality) -------------------------------------
            # (vr_i-vi_j)^2 + (vr_j+vi_i)^2 + 4*s_ij
            #   <= xij_s + 2*(vr_i+vi_j)*(vr0_i+vi0_j)
            #            + 2*(vr_j-vi_i)*(vr0_j-vi0_i)   <- note sign
            #            - ((vr0_i+vi0_j)^2 + (vr0_j-vi0_i)^2)
            rhs_4f1 = (
                xij_s
                + 2.0 * (vr_i + vi_j) * (vr0_i + vi0_j)
                + 2.0 * (vr_j - vi_i) * (vr0_j - vi0_i)
                - ((vr0_i + vi0_j) ** 2 + (vr0_j - vi0_i) ** 2)
            )
            m.addQConstr(
                (vr_i - vi_j) * (vr_i - vi_j)
                + (vr_j + vi_i) * (vr_j + vi_i)
                + 4.0 * s_ij <= rhs_4f1,
                name=f"4f1_{fi}_{ti}_{t}",
            )

            # ---- 4f (second inequality) ------------------------------------
            # (vr_i+vi_j)^2 + (vr_j-vi_i)^2 - 4*s_ij
            #   <= xij_s + 2*(vr_i-vi_j)*(vr0_i-vi0_j)
            #            + 2*(vr_j+vi_i)*(vr0_j+vi0_i)
            #            - ((vr0_i-vi0_j)^2 + (vr0_j+vi0_i)^2)
            rhs_4f2 = (
                xij_s
                + 2.0 * (vr_i - vi_j) * (vr0_i - vi0_j)
                + 2.0 * (vr_j + vi_i) * (vr0_j + vi0_i)
                - ((vr0_i - vi0_j) ** 2 + (vr0_j + vi0_i) ** 2)
            )
            m.addQConstr(
                (vr_i + vi_j) * (vr_i + vi_j)
                + (vr_j - vi_i) * (vr_j - vi_i)
                - 4.0 * s_ij <= rhs_4f2,
                name=f"4f2_{fi}_{ti}_{t}",
            )


# ---------------------------------------------------------------------------
# _add_mincost_obj_gurobi
# ---------------------------------------------------------------------------

def _add_mincost_obj_gurobi(
    m: gp.Model,
    data: MatpowerData,
    T: List[int],
    convex: bool = False,
    penalty_weight: float = 100000.0 # FIX: Accept the dynamic weight
) -> None:  
    """
    Set the min-cost quadratic objective on a gurobipy model.

    Non-convex:  sum_t sum_g  cost[0]*pg^2 + cost[1]*pg + cost[2]*u
    Convex:      same  +  sum_t (sum_i xi_c[i,t] + sum_e xij_c[e,t] + xij_s[e,t])
    
    """
    gens   = data.gens
    gen_ids = list(gens.keys())

    obj = gp.QuadExpr()
    for g in gen_ids:
        c = gens[g]['cost']
        for t in T:
            pg = m._pg[g, t]
            u  = m._u[g, t]
            obj += c[0] * pg * pg + c[1] * pg + c[2] * u

    if convex:
        # FIX: Use the injected penalty_weight instead of a hardcoded rho
        edges = get_edges(data)
        for i in data.buses:
            for t in T:
                obj += penalty_weight * m._xi_c[i, t]
        for (fi, ti) in edges:
            for t in T:
                obj += penalty_weight * (m._xij_c[fi, ti, t] + m._xij_s[fi, ti, t])

    m.setObjective(obj, GRB.MINIMIZE)

# ---------------------------------------------------------------------------
# _add_ref_limits_rectangular
# ---------------------------------------------------------------------------

def _add_ref_limits_rectangular(
    m: gp.Model,
    data: MatpowerData,
    T: List[int],
) -> None:
    """
    Reference bus: vi = 0, vr >= 0  (keeps solution in positive real half-plane).
    Mirrors Julia's _add_ref_limits_rectangular!().
    """
    ref_buses = [k for k, v in data.buses.items() if v['bus_type'] == 3]
    for i in ref_buses:
        for t in T:
            m.addConstr(m._vi[i, t] == 0.0, name=f"ref_vi_{i}_{t}")
            m.addConstr(m._vr[i, t] >= 0.0, name=f"ref_vr_{i}_{t}")


# ---------------------------------------------------------------------------
# _add_gen_limits_gurobi
# ---------------------------------------------------------------------------

def _add_gen_limits_gurobi(
    m: gp.Model,
    data: MatpowerData,
    T: List[int],
) -> None:
    """
    Generator operational limits tied to commitment status u:
        pmin*u <= pg <= pmax*u
        qmin*u <= qg <= qmax*u
    Mirrors Julia's _add_gen_limits!().
    """
    for g, gen in data.gens.items():
        for t in T:
            u  = m._u[g, t]
            pg = m._pg[g, t]
            qg = m._qg[g, t]
            m.addConstr(pg >= gen['pmin'] * u, name=f"pmin_{g}_{t}")
            m.addConstr(pg <= gen['pmax'] * u, name=f"pmax_{g}_{t}")
            m.addConstr(qg >= gen['qmin'] * u, name=f"qmin_{g}_{t}")
            m.addConstr(qg <= gen['qmax'] * u, name=f"qmax_{g}_{t}")


# ---------------------------------------------------------------------------
# _add_rectangular_branchflow
# ---------------------------------------------------------------------------

def _add_rectangular_branchflow(
    m: gp.Model,
    data: MatpowerData,
    T: List[int],
) -> None:
    """
    AC power flow in the W-matrix (rectangular) form (linear in c_ii, c_ij, s_ij)
    plus quadratic thermal limits.

    Note: p_to / q_to use c_ii[f_bus] to faithfully mirror Julia's formulation.jl
    (the standard formula would use c_ii[t_bus] for the to-side; the Julia code
    uses the from-bus c_ii for both sides – replicated here as-is).

    Mirrors Julia's _add_rectangular_branchflow!().
    """
    for k, branch in data.branches.items():
        fi    = branch['f_bus']
        ti    = branch['t_bus']
        f_str = str(fi)
        t_str = str(ti)

        g, b     = calc_branch_y(branch)
        tr, ti_  = calc_branch_t(branch)
        g_fr     = branch['g_fr']
        b_fr     = branch['b_fr']
        g_to     = branch['g_to']
        b_to     = branch['b_to']
        tm2      = branch['tap'] ** 2
        rate_a   = branch['rate_a']

        for t in T:
            c_ii = m._c_ii[f_str, t]
            c_ij = m._c_ij[fi, ti, t]
            s_ij = m._s_ij[fi, ti, t]
            p_fr = m._p_fr[k, t]
            q_fr = m._q_fr[k, t]
            p_to = m._p_to[k, t]
            q_to = m._q_to[k, t]
            c_jj = m._c_ii[t_str, t]   # <--- ADD THIS

            # From-side active power
            m.addConstr(
                p_fr == (g + g_fr) / tm2 * c_ii
                       + (-g * tr + b * ti_)  / tm2 * c_ij
                       + (-b * tr - g * ti_)  / tm2 * s_ij,
                name=f"pfr_{k}_{t}",
            )
            # From-side reactive power
            m.addConstr(
                q_fr == -(b + b_fr) / tm2 * c_ii
                        - (-b * tr - g * ti_) / tm2 * c_ij
                        + (-g * tr + b * ti_) / tm2 * s_ij,
                name=f"qfr_{k}_{t}",
            )
            # To-side active power
            m.addConstr(
                p_to == (g + g_to) * c_jj  # <--- CHANGE c_ii to c_jj
                       + (-g * tr - b * ti_) / tm2 * c_ij
                       + (-b * tr + g * ti_) / tm2 * s_ij,
                name=f"pto_{k}_{t}",
            )
            # To-side reactive power
            m.addConstr(
                q_to == -(b + b_to) * c_jj  # <--- CHANGE c_ii to c_jj
                        - (-b * tr + g * ti_) / tm2 * c_ij
                        + (-g * tr - b * ti_) / tm2 * s_ij,
                name=f"qto_{k}_{t}",
            )
            # Thermal limits  p^2 + q^2 <= rate_a^2
            m.addQConstr(
                p_fr * p_fr + q_fr * q_fr <= rate_a ** 2,
                name=f"thermal_fr_{k}_{t}",
            )
            m.addQConstr(
                p_to * p_to + q_to * q_to <= rate_a ** 2,
                name=f"thermal_to_{k}_{t}",
            )


# ---------------------------------------------------------------------------
# _add_node_bal_rectangular
# ---------------------------------------------------------------------------

def _precompute_bus_connectivity(data: MatpowerData) -> Dict:
    """
    Pre-compute per-bus connectivity (gens, loads, branch from/to) once.
    Avoids O(buses × components) list-comprehensions on every model build.
    Returns a dict keyed by bus_id with sub-dicts: bus_gens, bus_loads, br_fr, br_to, gs, bs.
    """
    buses    = data.buses
    gens     = data.gens
    branches = data.branches
    loads    = data.loads
    shunts   = data.shunts

    conn: Dict[str, Dict] = {}
    for i in buses:
        conn[i] = {
            'bus_gens':  [g for g, d in gens.items()     if d['gen_bus']         == i],
            'bus_loads': [l for l in loads.values()       if l['load_bus']        == i],
            'br_fr':     [k for k, b in branches.items() if str(b['f_bus'])       == i],
            'br_to':     [k for k, b in branches.items() if str(b['t_bus'])       == i],
            'gs':        sum(s['gs'] for s in shunts.values() if s['shunt_bus']   == i),
            'bs':        sum(s['bs'] for s in shunts.values() if s['shunt_bus']   == i),
            'base_pd':   sum(l['pd'] for l in loads.values() if l['load_bus']     == i),
            'base_qd':   sum(l['qd'] for l in loads.values() if l['load_bus']     == i),
        }
    return conn


def _add_node_bal_rectangular(
    m: gp.Model,
    data: MatpowerData,
    T: List[int],
    demand_curve: Optional[List[float]] = None,
    param_loads: Optional[Dict] = None,
    conn: Optional[Dict] = None,
) -> None:
    """
    Nodal power balance (Kirchhoff's current law) in rectangular form.

    demand_curve  – per-period load scale factors (length = len(T)).
                    If None, defaults to [1.0] * len(T) (no scaling).
    param_loads   – {'pd': {bus_id: float}, 'qd': {bus_id: float}}
                    If provided, overrides demand_curve for this solve.
    conn          – pre-computed connectivity dict from _precompute_bus_connectivity().
                    If None, computed on the fly (slower).

    Stores constraint references on the model:
        m._pbal[bus_id, t]  – active balance constraint  (RHS = pd)
        m._qbal[bus_id, t]  – reactive balance constraint (RHS = qd)
    These allow fast RHS updates without rebuilding the model.

    Mirrors Julia's _add_node_bal_rectangular!(...; param_loads=nothing).
    """
    buses = data.buses

    if demand_curve is None:
        demand_curve = [1.0] * len(T)

    if conn is None:
        conn = _precompute_bus_connectivity(data)

    m._pbal: Dict = {}
    m._qbal: Dict = {}

    for t_idx, t in enumerate(T):
        val = demand_curve[t_idx]

        for i in buses:
            c = conn[i]
            bus_gens = c['bus_gens']
            br_fr    = c['br_fr']
            br_to    = c['br_to']
            gs       = c['gs']
            bs       = c['bs']

            if param_loads is not None and param_loads.get('pd') is not None:
                pd = param_loads['pd'].get(i, 0.0)
                qd = param_loads['qd'].get(i, 0.0)
            else:
                pd = c['base_pd'] * val
                qd = c['base_qd'] * val

            c_ii = m._c_ii[i, t]

            # Active power balance: sum(pg) - gs*c_ii - sum(p_fr) - sum(p_to) = pd
            # RHS = pd  →  stored on m._pbal[i, t].RHS for fast parametric updates
            m._pbal[i, t] = m.addConstr(
                gp.quicksum(m._pg[g, t] for g in bus_gens)
                - gs * c_ii
                - gp.quicksum(m._p_fr[k, t] for k in br_fr)
                - gp.quicksum(m._p_to[k, t] for k in br_to)
                == pd,
                name=f"pbal_{i}_{t}",
            )
            # Reactive power balance: sum(qg) + bs*c_ii - sum(q_fr) - sum(q_to) = qd
            m._qbal[i, t] = m.addConstr(
                gp.quicksum(m._qg[g, t] for g in bus_gens)
                + bs * c_ii
                - gp.quicksum(m._q_fr[k, t] for k in br_fr)
                - gp.quicksum(m._q_to[k, t] for k in br_to)
                == qd,
                name=f"qbal_{i}_{t}",
            )


# ===========================================================================
# ── PYOMO + IPOPT  (Polar / NL formulations) ────────────────────────────────
# ===========================================================================

# ---------------------------------------------------------------------------
# build_single_period_ac_uc_polar
# ---------------------------------------------------------------------------

def build_single_period_ac_uc_polar(data: MatpowerData) -> pyo.ConcreteModel:
    """
    Single-period AC-UC in polar coordinates (NL AC power flow).

    Solver: Ipopt  (u relaxed to continuous [0,1] for NLP compatibility)
    Mirrors Julia's build_single_period_ac_uc_polar(file_path).
    """
    T = [1]
    m = pyo.ConcreteModel(name="single_ac_uc_polar")

    _add_acuc_var_polar(m, data, T)
    _add_mincost_obj_pyomo(m, data, T)
    _add_ref_limits_polar(m, data, T)
    _add_gen_limits_pyomo(m, data, T)
    _add_polar_branchflow(m, data, T)
    _add_node_bal_polar(m, data, T)
    return m


# ---------------------------------------------------------------------------
# build_mp_ac_uc_polar
# ---------------------------------------------------------------------------

def build_mp_ac_uc_polar(
    data: MatpowerData,
    demand_curve: List[float],
) -> pyo.ConcreteModel:
    """
    Multi-period AC-UC in polar coordinates (NL AC power flow).

    Mirrors Julia's build_mp_ac_uc_polar(file_path, demand_curve).
    """
    T = list(range(1, len(demand_curve) + 1))
    m = pyo.ConcreteModel(name="mp_ac_uc_polar")

    _add_acuc_var_polar(m, data, T)
    _add_mincost_obj_pyomo(m, data, T)
    _add_ref_limits_polar(m, data, T)
    _add_gen_limits_pyomo(m, data, T)
    _add_polar_branchflow(m, data, T)
    _add_node_bal_polar(m, data, T, demand_curve=demand_curve)
    return m


def solve_polar(model: pyo.ConcreteModel, tee: bool = False) -> None:
    """Solve a polar pyomo model with Ipopt."""
    solver = SolverFactory('ipopt')
    result = solver.solve(model, tee=tee)
    return result


# ---------------------------------------------------------------------------
# _add_acuc_var_polar
# ---------------------------------------------------------------------------

def _add_acuc_var_polar(
    m: pyo.ConcreteModel,
    data: MatpowerData,
    T: List[int],
) -> None:
    """
    Add polar-coordinate AC-UC variables to a pyomo model.
    Mirrors Julia's _add_acuc_var_polar!().
    """
    buses    = data.buses
    gens     = data.gens
    branches = data.branches

    bus_ids    = list(buses.keys())
    gen_ids    = list(gens.keys())
    branch_ids = list(branches.keys())

    m.T          = pyo.Set(initialize=T)
    m.bus_ids    = pyo.Set(initialize=bus_ids)
    m.gen_ids    = pyo.Set(initialize=gen_ids)
    m.branch_ids = pyo.Set(initialize=branch_ids)

    # Voltage angle (unbounded)
    m.va = pyo.Var(m.bus_ids, m.T, domain=pyo.Reals, initialize=0.0)

    # Voltage magnitude  [vmin, vmax]
    m.vm = pyo.Var(
        m.bus_ids, m.T,
        domain=pyo.NonNegativeReals,
        bounds=lambda mdl, i, t: (buses[i]['vmin'], buses[i]['vmax']),
        initialize=1.0,
    )

    # Unit commitment – relaxed to [0,1] for Ipopt
    m.u  = pyo.Var(m.gen_ids, m.T, domain=pyo.UnitInterval,   initialize=1.0)
    m.pg = pyo.Var(m.gen_ids, m.T, domain=pyo.Reals,          initialize=0.0)
    m.qg = pyo.Var(m.gen_ids, m.T, domain=pyo.Reals,          initialize=0.0)

    # Branch flow variables
    m.p_fr = pyo.Var(m.branch_ids, m.T, domain=pyo.Reals, initialize=0.0)
    m.q_fr = pyo.Var(m.branch_ids, m.T, domain=pyo.Reals, initialize=0.0)
    m.p_to = pyo.Var(m.branch_ids, m.T, domain=pyo.Reals, initialize=0.0)
    m.q_to = pyo.Var(m.branch_ids, m.T, domain=pyo.Reals, initialize=0.0)


# ---------------------------------------------------------------------------
# _add_mincost_obj_pyomo
# ---------------------------------------------------------------------------

def _add_mincost_obj_pyomo(
    m: pyo.ConcreteModel,
    data: MatpowerData,
    T: List[int],
) -> None:
    """
    Min-cost objective for a pyomo model.
    Mirrors Julia's _add_mincost_obj!() (non-convex path).
    """
    gens    = data.gens
    gen_ids = list(gens.keys())

    m.obj = pyo.Objective(
        expr=sum(
            gens[g]['cost'][0] * m.pg[g, t] ** 2
            + gens[g]['cost'][1] * m.pg[g, t]
            + gens[g]['cost'][2] * m.u[g, t]
            for g in gen_ids for t in T
        ),
        sense=pyo.minimize,
    )


# ---------------------------------------------------------------------------
# _add_ref_limits_polar
# ---------------------------------------------------------------------------

def _add_ref_limits_polar(
    m: pyo.ConcreteModel,
    data: MatpowerData,
    T: List[int],
) -> None:
    """
    Reference bus angle = 0.
    Mirrors Julia's _add_ref_limits_polar!().
    """
    ref_buses = [k for k, v in data.buses.items() if v['bus_type'] == 3]

    def ref_va_rule(mdl, i, t):
        if i in ref_buses:
            return mdl.va[i, t] == 0.0
        return pyo.Constraint.Skip

    m.ref_va = pyo.Constraint(m.bus_ids, m.T, rule=ref_va_rule)


# ---------------------------------------------------------------------------
# _add_gen_limits_pyomo
# ---------------------------------------------------------------------------

def _add_gen_limits_pyomo(
    m: pyo.ConcreteModel,
    data: MatpowerData,
    T: List[int],
) -> None:
    """
    pmin*u <= pg <= pmax*u,  qmin*u <= qg <= qmax*u.
    Mirrors Julia's _add_gen_limits!().
    """
    gens = data.gens

    m.pmin_con = pyo.Constraint(
        m.gen_ids, m.T,
        rule=lambda mdl, g, t: mdl.pg[g, t] >= gens[g]['pmin'] * mdl.u[g, t],
    )
    m.pmax_con = pyo.Constraint(
        m.gen_ids, m.T,
        rule=lambda mdl, g, t: mdl.pg[g, t] <= gens[g]['pmax'] * mdl.u[g, t],
    )
    m.qmin_con = pyo.Constraint(
        m.gen_ids, m.T,
        rule=lambda mdl, g, t: mdl.qg[g, t] >= gens[g]['qmin'] * mdl.u[g, t],
    )
    m.qmax_con = pyo.Constraint(
        m.gen_ids, m.T,
        rule=lambda mdl, g, t: mdl.qg[g, t] <= gens[g]['qmax'] * mdl.u[g, t],
    )


# ---------------------------------------------------------------------------
# _add_polar_branchflow
# ---------------------------------------------------------------------------

def _add_polar_branchflow(
    m: pyo.ConcreteModel,
    data: MatpowerData,
    T: List[int],
) -> None:
    """
    AC power flow equations in polar coordinates (NL: sin / cos).
    Thermal limits as quadratic inequalities.
    Mirrors Julia's _add_polar_branchflow!() (which used @NLconstraint + Ipopt).
    """
    branches = data.branches

    def pfr_rule(mdl, k, t):
        br   = branches[k]
        f    = str(br['f_bus'])
        tb   = str(br['t_bus'])
        g, b = calc_branch_y(br)
        tr, ti = calc_branch_t(br)
        g_fr = br['g_fr'];  b_fr = br['b_fr']
        tm2  = br['tap'] ** 2
        return mdl.p_fr[k, t] == (
            (g + g_fr) / tm2 * mdl.vm[f, t] ** 2
            + (-g * tr + b * ti)  / tm2 * mdl.vm[f, t] * mdl.vm[tb, t]
              * pyo.cos(mdl.va[f, t] - mdl.va[tb, t])
            + (-b * tr - g * ti)  / tm2 * mdl.vm[f, t] * mdl.vm[tb, t]
              * pyo.sin(mdl.va[f, t] - mdl.va[tb, t])
        )

    def qfr_rule(mdl, k, t):
        br   = branches[k]
        f    = str(br['f_bus'])
        tb   = str(br['t_bus'])
        g, b = calc_branch_y(br)
        tr, ti = calc_branch_t(br)
        b_fr = br['b_fr']
        tm2  = br['tap'] ** 2
        return mdl.q_fr[k, t] == (
            -(b + b_fr) / tm2 * mdl.vm[f, t] ** 2
            - (-b * tr - g * ti) / tm2 * mdl.vm[f, t] * mdl.vm[tb, t]
              * pyo.cos(mdl.va[f, t] - mdl.va[tb, t])
            + (-g * tr + b * ti) / tm2 * mdl.vm[f, t] * mdl.vm[tb, t]
              * pyo.sin(mdl.va[f, t] - mdl.va[tb, t])
        )

    def pto_rule(mdl, k, t):
        br   = branches[k]
        f    = str(br['f_bus'])
        tb   = str(br['t_bus'])
        g, b = calc_branch_y(br)
        tr, ti = calc_branch_t(br)
        g_to = br['g_to']
        tm2  = br['tap'] ** 2
        return mdl.p_to[k, t] == (
            (g + g_to) * mdl.vm[tb, t] ** 2
            + (-g * tr - b * ti) / tm2 * mdl.vm[tb, t] * mdl.vm[f, t]
              * pyo.cos(mdl.va[tb, t] - mdl.va[f, t])
            + (-b * tr + g * ti) / tm2 * mdl.vm[tb, t] * mdl.vm[f, t]
              * pyo.sin(mdl.va[tb, t] - mdl.va[f, t])
        )

    def qto_rule(mdl, k, t):
        br   = branches[k]
        f    = str(br['f_bus'])
        tb   = str(br['t_bus'])
        g, b = calc_branch_y(br)
        tr, ti = calc_branch_t(br)
        b_to = br['b_to']
        tm2  = br['tap'] ** 2
        return mdl.q_to[k, t] == (
            -(b + b_to) * mdl.vm[tb, t] ** 2
            - (-b * tr + g * ti) / tm2 * mdl.vm[tb, t] * mdl.vm[f, t]
              * pyo.cos(mdl.va[tb, t] - mdl.va[f, t])
            + (-g * tr - b * ti) / tm2 * mdl.vm[tb, t] * mdl.vm[f, t]
              * pyo.sin(mdl.va[tb, t] - mdl.va[f, t])
        )

    def thermal_fr_rule(mdl, k, t):
        return mdl.p_fr[k, t] ** 2 + mdl.q_fr[k, t] ** 2 \
               <= branches[k]['rate_a'] ** 2

    def thermal_to_rule(mdl, k, t):
        return mdl.p_to[k, t] ** 2 + mdl.q_to[k, t] ** 2 \
               <= branches[k]['rate_a'] ** 2

    m.pfr_con        = pyo.Constraint(m.branch_ids, m.T, rule=pfr_rule)
    m.qfr_con        = pyo.Constraint(m.branch_ids, m.T, rule=qfr_rule)
    m.pto_con        = pyo.Constraint(m.branch_ids, m.T, rule=pto_rule)
    m.qto_con        = pyo.Constraint(m.branch_ids, m.T, rule=qto_rule)
    m.thermal_fr_con = pyo.Constraint(m.branch_ids, m.T, rule=thermal_fr_rule)
    m.thermal_to_con = pyo.Constraint(m.branch_ids, m.T, rule=thermal_to_rule)


# ---------------------------------------------------------------------------
# _add_node_bal_polar
# ---------------------------------------------------------------------------

def _add_node_bal_polar(
    m: pyo.ConcreteModel,
    data: MatpowerData,
    T: List[int],
    demand_curve: Optional[List[float]] = None,
) -> None:
    """
    Nodal power balance in polar coordinates.
    Mirrors Julia's _add_node_bal_polar!().
    """
    buses    = data.buses
    gens     = data.gens
    branches = data.branches
    loads    = data.loads
    shunts   = data.shunts

    if demand_curve is None:
        demand_curve = [1.0] * len(T)

    # Pre-compute per-bus load and shunt for each period
    def pbal_rule(mdl, i, t):
        t_idx     = T.index(t)
        val       = demand_curve[t_idx]
        bus_loads = [l for l in loads.values()  if l['load_bus'] == i]
        bus_gens  = [g for g, d in gens.items()  if d['gen_bus'] == i]
        br_fr     = [k for k, b in branches.items() if str(b['f_bus']) == i]
        br_to     = [k for k, b in branches.items() if str(b['t_bus']) == i]

        pd = sum(l['pd'] for l in bus_loads) * val
        gs = sum(s['gs'] for s in shunts.values() if s['shunt_bus'] == i)

        return (
            sum(mdl.pg[g, t] for g in bus_gens)
            - pd - gs * mdl.vm[i, t] ** 2
            == sum(mdl.p_fr[k, t] for k in br_fr)
               + sum(mdl.p_to[k, t] for k in br_to)
        )

    def qbal_rule(mdl, i, t):
        t_idx     = T.index(t)
        val       = demand_curve[t_idx]
        bus_loads = [l for l in loads.values()  if l['load_bus'] == i]
        bus_gens  = [g for g, d in gens.items()  if d['gen_bus'] == i]
        br_fr     = [k for k, b in branches.items() if str(b['f_bus']) == i]
        br_to     = [k for k, b in branches.items() if str(b['t_bus']) == i]

        qd = sum(l['qd'] for l in bus_loads) * val
        bs = sum(s['bs'] for s in shunts.values() if s['shunt_bus'] == i)

        return (
            sum(mdl.qg[g, t] for g in bus_gens)
            - qd + bs * mdl.vm[i, t] ** 2
            == sum(mdl.q_fr[k, t] for k in br_fr)
               + sum(mdl.q_to[k, t] for k in br_to)
        )

    m.pbal_con = pyo.Constraint(m.bus_ids, m.T, rule=pbal_rule)
    m.qbal_con = pyo.Constraint(m.bus_ids, m.T, rule=qbal_rule)


# ===========================================================================
# Training data generation  (mirrors generate_training_data in test.ipynb)
# ===========================================================================

def generate_training_data(
    data: MatpowerData,
    node_vr: Dict[str, float],
    node_vi: Dict[str, float],
    n_samples: int = 50,
    load_scale_min: float = 0.6,
    load_scale_max: float = 1.4,
    seed: int = 42,
    silent: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate training samples by varying loads and solving the convex AC-UC.

    Speed-critical design
    ---------------------
    The Gurobi model is built ONCE.  For each sample only the node-balance
    constraint RHS values (pd_i, qd_i) are updated via constr.RHS = new_value
    before re-solving.  This avoids 50× full Python model-build overhead and
    gives Julia-comparable throughput.

    Data Generation Strategy (Macro Scale + Micro Noise)
    ----------------------------------------------------
    To prevent the neural network from overfitting to a simple 1D demand curve 
    while maintaining physically realistic grid states, the load scaling combines 
    a global system trend with independent localized jitter.

    For each sample:
        1. Sample a global macro-scale factor uniformly from [load_scale_min, load_scale_max].
        2. Sample independent micro-noise (e.g., Gaussian) for each node's Pd and Qd.
        3. Combine the global scale and local noise to compute the final nodal demands.
        4. Update pbal/qbal constraint RHS (O(num_buses) — negligible).
        5. Re-solve with warm-start from the previous solution.
        6. If optimal: record (loads, pg, u).

    Returns
    -------
    X     : (n_valid, 2*num_buses)   [pd_1, qd_1, pd_2, qd_2, ...] sorted bus order
    Y_pg  : (n_valid, num_gens)      active power outputs (pu)
    Y_u   : (n_valid, num_gens)      binary commitment decisions

    Mirrors generate_training_data() in test.ipynb.
    """
    import time
    rng = np.random.default_rng(seed)

    buses          = data.buses
    gens           = data.gens
    bus_ids_sorted = sorted(buses.keys())
    gen_ids_sorted = sorted(gens.keys())

    # ── Pre-compute connectivity once (avoids O(n²) scans per sample) ───────
    conn = _precompute_bus_connectivity(data)
    base_pd = {i: conn[i]['base_pd'] for i in buses}
    base_qd = {i: conn[i]['base_qd'] for i in buses}

    # ── Build the model ONCE with baseline loads (scale = 1.0) ───────────────
    print("  Building convex AC-UC model (once) ...")
    t0 = time.time()
    mdl = build_convex_ac_uc(
        data, node_vr, node_vi,
        node_pd={i: base_pd[i] for i in buses},
        node_qd={i: base_qd[i] for i in buses},
        conn=conn,
    )
    mdl.setParam('OutputFlag', 0)
    mdl.setParam('MIPGap',     1e-3)   # relaxed for training data (exact optimality not needed)
    mdl.setParam('TimeLimit',  10.0)   # per-sample safety net
    # Relax barrier tolerances to handle borderline numerical issues
    mdl.setParam('BarConvTol',      1e-6)
    mdl.setParam('FeasibilityTol',  1e-4)
    mdl.setParam('OptimalityTol',   1e-4)
    mdl.setParam('BarIterLimit',    200)  # prevent barrier from spinning
    mdl.update()
    print(f"  Model built in {time.time()-t0:.2f}s  "
          f"({mdl.NumVars} vars, {mdl.NumConstrs} linear + {mdl.NumQConstrs} quad constrs)")

    # ── Initial solve at baseline to establish warm-start ────────────────────
    mdl.optimize()
    if mdl.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        print(f"  Baseline solve OK (obj={mdl.ObjVal:.4f})")
    else:
        print(f"  WARNING: baseline solve status {mdl.status}")

    # ── Parametric loop: update RHS only, re-solve ────────────────────────────
    X_list:   list = []
    Ypg_list: list = []
    Yu_list:  list = []

    t_start = time.time()
    attempts = 0
    
    # CHANGE: Use a while loop to guarantee we get exactly n_samples
    while len(X_list) < n_samples:
        attempts += 1
        
        # 1. The Macro Trend (Time of Day)
        base_scale = rng.uniform(load_scale_min, load_scale_max)
        
        current_pd = {}
        current_qd = {}
        
        for i in buses:
            # 2. The Micro Noise (Reduce scale to 0.02 if you want fewer failures)
            nodal_noise_p = rng.normal(loc=1.0, scale=0.05) 
            nodal_noise_q = rng.normal(loc=1.0, scale=0.05) 
            
            current_pd[i] = base_pd[i] * base_scale * nodal_noise_p
            current_qd[i] = base_qd[i] * base_scale * nodal_noise_q
            
            mdl._pbal[i, 1].RHS = current_pd[i]
            mdl._qbal[i, 1].RHS = current_qd[i]
            
        mdl.update()
        mdl.optimize()

        if mdl.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            x_vec    = [v for i in bus_ids_sorted for v in (current_pd[i], current_qd[i])]
            y_pg_vec = [mdl._pg[g, 1].X       for g in gen_ids_sorted]
            y_u_vec  = [round(mdl._u[g, 1].X) for g in gen_ids_sorted]

            X_list.append(x_vec)
            Ypg_list.append(y_pg_vec)
            Yu_list.append(y_u_vec)
            
            if not silent and len(X_list) % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  {len(X_list)}/{n_samples} feasible samples collected... "
                      f"(Attempt {attempts}, {elapsed:.1f}s elapsed)")
        else:
            if not silent:
                print(f"  Sample {s+1}: solver status {status} – skipped.")

        if not silent and (s + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  {s+1}/{n_samples} samples  ({elapsed:.1f}s elapsed, "
                  f"{elapsed/(s+1):.2f}s/sample)")

    if not X_list:
        raise RuntimeError("No feasible samples generated. Check warm-start solution.")

    X    = np.array(X_list,   dtype=np.float32)
    Y_pg = np.array(Ypg_list, dtype=np.float32)
    Y_u  = np.array(Yu_list,  dtype=np.float32)

    print(f"  Done: {len(X_list)}/{n_samples} feasible samples in "
          f"{time.time()-t_start:.1f}s  ({(time.time()-t_start)/max(len(X_list),1):.2f}s/sample)")
    return X, Y_pg, Y_u


# ===========================================================================
# Ground-truth MIQCQP solver  (batch + single-sample API)
# ===========================================================================

class GroundTruthSolver:
    """
    Reusable ground-truth solver that builds the convex QCQP model ONCE
    and updates only the node-balance RHS for each new load profile.

    Uses the convex relaxation (fast barrier) with relaxed tolerances
    to handle borderline instances.

    Usage
    -----
        solver = GroundTruthSolver(data, node_vr, node_vi)
        u, pg, obj = solver.solve(pd_dict, qd_dict)   # fast per-sample
    """

    def __init__(
        self,
        data: MatpowerData,
        node_vr: Dict[str, float],
        node_vi: Dict[str, float],
    ) -> None:
        self._data = data
        self._gen_ids_sorted = sorted(data.gens.keys())
        self._bus_ids = list(data.buses.keys())

        conn = _precompute_bus_connectivity(data)
        base_pd = {i: conn[i]['base_pd'] for i in data.buses}
        base_qd = {i: conn[i]['base_qd'] for i in data.buses}

        self._mdl = build_convex_ac_uc(
            data, node_vr, node_vi, base_pd, base_qd, conn=conn,
        )
        self._mdl.setParam('OutputFlag', 0)
        self._mdl.setParam('TimeLimit', 3.0)    # fail fast on hard instances
        self._mdl.setParam('MIPGap', 1e-4)
        # Relax barrier tolerances to handle borderline numerical issues
        self._mdl.setParam('BarConvTol', 1e-6)
        self._mdl.setParam('FeasibilityTol', 1e-4)
        self._mdl.setParam('OptimalityTol', 1e-4)

    def solve(
        self,
        pd: Dict[str, float],
        qd: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve for a given load profile by updating RHS only."""
        m = self._mdl
        for i in self._bus_ids:
            m._pbal[i, 1].RHS = pd.get(i, 0.0)
            m._qbal[i, 1].RHS = qd.get(i, 0.0)
        m.update()
        m.optimize()

        if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            u = np.array([round(m._u[g, 1].X) for g in self._gen_ids_sorted])
            pg = np.array([m._pg[g, 1].X for g in self._gen_ids_sorted])
            return u, pg, m.ObjVal

        n = len(self._gen_ids_sorted)
        return np.zeros(n), np.zeros(n), float('inf')


def solve_true_miqcqp_for_sample(
    data: MatpowerData,
    node_vr: Dict[str, float],
    node_vi: Dict[str, float],
    pd: Dict[str, float],
    qd: Dict[str, float],
    _solver_cache: Dict = {},
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve the full convex QCQP (with binary u) for a given load profile.

    Builds the Gurobi model on first call and caches it for subsequent calls
    (parametric RHS update only — same speed as generate_training_data).

    Parameters
    ----------
    data    : parsed MATPOWER network
    node_vr : warm-start real voltages
    node_vi : warm-start imaginary voltages
    pd, qd  : per-bus load dicts (bus_str -> float, pu)

    Returns
    -------
    u_true  : (n_gen,) binary commitment
    pg_true : (n_gen,) active power (pu)
    obj     : scalar objective value
    """
    cache_key = id(data)
    if cache_key not in _solver_cache:
        _solver_cache[cache_key] = GroundTruthSolver(data, node_vr, node_vi)
    return _solver_cache[cache_key].solve(pd, qd)
