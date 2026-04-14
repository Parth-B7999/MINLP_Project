"""
cvxpy_formulation.py
====================
CVXPY DCP-compliant translation of the convex AC-UC QCQP relaxation
(Constante-Flores & Li 2026) wrapped as a CvxpyLayer for differentiable
self-supervised training.

Mirrors build_convex_ac_uc() from formulation.py but expressed entirely in
CVXPY so that torch gradients can flow through the KKT system via cvxpylayers.

Key design decisions
--------------------
- All variables are indexed by integer position (0-based numpy arrays) since
  CVXPY operates on dense vectors, not dict-keyed variables.
- Index mappings (bus_to_idx, gen_to_idx, branch_to_idx, edge_to_idx) are
  stored in an ACUCConfig dataclass returned alongside the CvxpyLayer.
- The CvxpyLayer parameters are: A_cut, b_cut (learned cuts on u),
  pd_param, qd_param (per-bus loads).
- u is relaxed from binary to continuous [0, 1] for differentiability.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .data_utils import MatpowerData, calc_branch_y, calc_branch_t, get_edges


# ---------------------------------------------------------------------------
# Config dataclass — stores index mappings and network parameters
# ---------------------------------------------------------------------------

@dataclass
class ACUCConfig:
    """Stores index mappings and network constants needed for restoration/eval."""
    bus_ids: List[str]          # sorted bus string ids
    gen_ids: List[str]          # sorted gen string ids
    branch_ids: List[str]       # sorted branch string ids
    edges: List[Tuple[int,int]] # (f_bus_int, t_bus_int) per branch

    bus_to_idx: Dict[str, int] = field(default_factory=dict)
    gen_to_idx: Dict[str, int] = field(default_factory=dict)
    branch_to_idx: Dict[str, int] = field(default_factory=dict)
    edge_to_idx: Dict[Tuple[int,int], int] = field(default_factory=dict)

    n_bus: int = 0
    n_gen: int = 0
    n_branch: int = 0
    n_edge: int = 0

    # Generator cost coefficients (arrays, gen-order)
    cost_c2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cost_c1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cost_c0: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Generator limits (arrays, gen-order)
    pmin: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pmax: np.ndarray = field(default_factory=lambda: np.zeros(0))
    qmin: np.ndarray = field(default_factory=lambda: np.zeros(0))
    qmax: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Bus voltage limits
    vmin: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vmax: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Reference bus index
    ref_bus_idx: int = 0

    # Per-bus connectivity for node balance
    bus_gen_map: Dict[int, List[int]] = field(default_factory=dict)
    bus_shunt_gs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bus_shunt_bs: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _build_config(data: MatpowerData) -> ACUCConfig:
    """Build index mappings and extract network constants as numpy arrays."""
    bus_ids = sorted(data.buses.keys())
    gen_ids = sorted(data.gens.keys())
    branch_ids = sorted(data.branches.keys())
    edges = get_edges(data)

    bus_to_idx = {b: i for i, b in enumerate(bus_ids)}
    gen_to_idx = {g: i for i, g in enumerate(gen_ids)}
    branch_to_idx = {k: i for i, k in enumerate(branch_ids)}
    edge_to_idx = {e: i for i, e in enumerate(edges)}

    n_bus = len(bus_ids)
    n_gen = len(gen_ids)
    n_branch = len(branch_ids)

    cfg = ACUCConfig(
        bus_ids=bus_ids, gen_ids=gen_ids, branch_ids=branch_ids, edges=edges,
        bus_to_idx=bus_to_idx, gen_to_idx=gen_to_idx,
        branch_to_idx=branch_to_idx, edge_to_idx=edge_to_idx,
        n_bus=n_bus, n_gen=n_gen, n_branch=n_branch, n_edge=len(edges),
    )

    # Cost coefficients
    cfg.cost_c2 = np.array([data.gens[g]['cost'][0] for g in gen_ids])
    cfg.cost_c1 = np.array([data.gens[g]['cost'][1] for g in gen_ids])
    cfg.cost_c0 = np.array([data.gens[g]['cost'][2] for g in gen_ids])

    # Generator limits
    cfg.pmin = np.array([data.gens[g]['pmin'] for g in gen_ids])
    cfg.pmax = np.array([data.gens[g]['pmax'] for g in gen_ids])
    cfg.qmin = np.array([data.gens[g]['qmin'] for g in gen_ids])
    cfg.qmax = np.array([data.gens[g]['qmax'] for g in gen_ids])

    # Bus limits
    cfg.vmin = np.array([data.buses[b]['vmin'] for b in bus_ids])
    cfg.vmax = np.array([data.buses[b]['vmax'] for b in bus_ids])

    # Reference bus
    ref_buses = [b for b in bus_ids if data.buses[b]['bus_type'] == 3]
    cfg.ref_bus_idx = bus_to_idx[ref_buses[0]]

    # Per-bus gen mapping (bus_idx -> list of gen_idx)
    cfg.bus_gen_map = {i: [] for i in range(n_bus)}
    for g in gen_ids:
        bus_str = data.gens[g]['gen_bus']
        bi = bus_to_idx[bus_str]
        gi = gen_to_idx[g]
        cfg.bus_gen_map[bi].append(gi)

    # Shunt data per bus
    cfg.bus_shunt_gs = np.zeros(n_bus)
    cfg.bus_shunt_bs = np.zeros(n_bus)
    for sid, sh in data.shunts.items():
        bi = bus_to_idx[sh['shunt_bus']]
        cfg.bus_shunt_gs[bi] += sh['gs']
        cfg.bus_shunt_bs[bi] += sh['bs']

    return cfg


# ---------------------------------------------------------------------------
# Main builder: CVXPY problem + CvxpyLayer
# ---------------------------------------------------------------------------

def build_cvxpy_ac_uc_layer(
    data: MatpowerData,
    node_vr: Dict[str, float],
    node_vi: Dict[str, float],
    n_cuts: int = 10,
) -> Tuple[CvxpyLayer, ACUCConfig]:
    """
    Build a CVXPY DCP-compliant convex AC-UC problem and wrap it as a
    CvxpyLayer for differentiable self-supervised training.

    Parameters
    ----------
    data      : parsed MATPOWER network
    node_vr   : warm-start real voltages (from rectangular MIQCQP solve)
    node_vi   : warm-start imaginary voltages
    n_cuts    : number of learned linear cuts on u (V_CONSTRAINTS)

    Returns
    -------
    layer : CvxpyLayer  with parameters [A_cut, b_cut, pd_param, qd_param]
            and variables  [u, pg, qg, vr, vi, c_ii, s_cut]
    cfg   : ACUCConfig   with index mappings and constants
    """
    cfg = _build_config(data)
    NB = cfg.n_bus
    NG = cfg.n_gen
    NK = cfg.n_branch
    NE = cfg.n_edge

    # ===== CVXPY Variables =====
    vr     = cp.Variable(NB, name="vr")
    vi     = cp.Variable(NB, name="vi")
    c_ii   = cp.Variable(NB, name="c_ii")
    c_ij   = cp.Variable(NE, name="c_ij")
    s_ij   = cp.Variable(NE, name="s_ij")
    u      = cp.Variable(NG, name="u")       # relaxed to [0,1]
    pg     = cp.Variable(NG, name="pg")
    qg     = cp.Variable(NG, name="qg")
    p_fr   = cp.Variable(NK, name="p_fr")
    q_fr   = cp.Variable(NK, name="q_fr")
    p_to   = cp.Variable(NK, name="p_to")
    q_to   = cp.Variable(NK, name="q_to")
    xi_c   = cp.Variable(NB, name="xi_c", nonneg=True)
    xij_c  = cp.Variable(NE, name="xij_c", nonneg=True)
    xij_s  = cp.Variable(NE, name="xij_s", nonneg=True)
    s_cut  = cp.Variable(n_cuts, name="s_cut", nonneg=True)

    # ===== CVXPY Parameters (NN-predicted + load inputs) =====
    A_cut    = cp.Parameter((n_cuts, NG), name="A_cut")
    b_cut    = cp.Parameter(n_cuts, name="b_cut")
    pd_param = cp.Parameter(NB, name="pd")
    qd_param = cp.Parameter(NB, name="qd")

    constraints = []

    # ----- u bounds [0, 1] -----
    constraints += [u >= 0, u <= 1]

    # ----- Generator limits: pmin*u <= pg <= pmax*u, same for qg -----
    for g_idx in range(NG):
        constraints += [
            pg[g_idx] >= cfg.pmin[g_idx] * u[g_idx],
            pg[g_idx] <= cfg.pmax[g_idx] * u[g_idx],
            qg[g_idx] >= cfg.qmin[g_idx] * u[g_idx],
            qg[g_idx] <= cfg.qmax[g_idx] * u[g_idx],
        ]

    # ----- Voltage bounds: c_ii in [vmin^2, vmax^2] -----
    constraints += [
        c_ii >= cfg.vmin ** 2,
        c_ii <= cfg.vmax ** 2,
    ]

    # ----- Reference bus: vi[ref] == 0, vr[ref] >= 0 -----
    ri = cfg.ref_bus_idx
    constraints += [vi[ri] == 0, vr[ri] >= 0]

    # ----- Convex constraints 4b, 4c (diagonal W-matrix) -----
    for i_str in cfg.bus_ids:
        idx = cfg.bus_to_idx[i_str]
        vr0 = node_vr[i_str]
        vi0 = node_vi[i_str]

        # 4b: c_ii >= vr^2 + vi^2  (SOC-representable, DCP: convex <= affine)
        constraints.append(
            cp.sum_squares(cp.hstack([vr[idx], vi[idx]])) <= c_ii[idx]
        )

        # 4c: c_ii <= 2*(vr0*vr + vi0*vi) - (vr0^2 + vi0^2) + xi_c
        constraints.append(
            c_ii[idx] <= 2.0 * (vr0 * vr[idx] + vi0 * vi[idx])
                         - (vr0**2 + vi0**2)
                         + xi_c[idx]
        )

    # ----- Convex constraints 4d, 4e, 4f1, 4f2 (off-diagonal W-matrix) -----
    for (fi, ti) in cfg.edges:
        e_idx = cfg.edge_to_idx[(fi, ti)]
        i_str = str(fi)
        j_str = str(ti)
        i_idx = cfg.bus_to_idx[i_str]
        j_idx = cfg.bus_to_idx[j_str]

        vr0_i = node_vr[i_str]; vr0_j = node_vr[j_str]
        vi0_i = node_vi[i_str]; vi0_j = node_vi[j_str]

        # --- 4d ---
        # (vr_i+vr_j)^2 + (vi_i+vi_j)^2 + 4*c_ij <= rhs_4d
        rhs_4d = (
            xij_c[e_idx]
            + 2.0 * (vr[i_idx] - vr[j_idx]) * (vr0_i - vr0_j)
            + 2.0 * (vi[i_idx] - vi[j_idx]) * (vi0_i - vi0_j)
            - ((vr0_i - vr0_j)**2 + (vi0_i - vi0_j)**2)
        )
        constraints.append(
            cp.sum_squares(cp.hstack([
                vr[i_idx] + vr[j_idx],
                vi[i_idx] + vi[j_idx],
            ])) + 4.0 * c_ij[e_idx] <= rhs_4d
        )

        # --- 4e ---
        rhs_4e = (
            xij_c[e_idx]
            + 2.0 * (vr[i_idx] + vr[j_idx]) * (vr0_i + vr0_j)
            + 2.0 * (vi[i_idx] + vi[j_idx]) * (vi0_i + vi0_j)
            - ((vr0_i + vr0_j)**2 + (vi0_i + vi0_j)**2)
        )
        constraints.append(
            cp.sum_squares(cp.hstack([
                vr[i_idx] - vr[j_idx],
                vi[i_idx] - vi[j_idx],
            ])) + 4.0 * c_ij[e_idx] <= rhs_4e
        )

        # --- 4f1 ---
        rhs_4f1 = (
            xij_s[e_idx]
            + 2.0 * (vr[i_idx] + vi[j_idx]) * (vr0_i + vi0_j)
            + 2.0 * (vr[j_idx] - vi[i_idx]) * (vr0_j + vi0_i)
            - ((vr0_i + vi0_j)**2 + (vr0_j - vi0_i)**2)
        )
        constraints.append(
            cp.sum_squares(cp.hstack([
                vr[i_idx] - vi[j_idx],
                vr[j_idx] + vi[i_idx],
            ])) + 4.0 * s_ij[e_idx] <= rhs_4f1
        )

        # --- 4f2 ---
        rhs_4f2 = (
            xij_s[e_idx]
            + 2.0 * (vr[i_idx] - vi[j_idx]) * (vr0_i - vi0_j)
            + 2.0 * (vr[j_idx] + vi[i_idx]) * (vr0_j + vi0_i)
            - ((vr0_i - vi0_j)**2 + (vr0_j + vi0_i)**2)
        )
        constraints.append(
            cp.sum_squares(cp.hstack([
                vr[i_idx] + vi[j_idx],
                vr[j_idx] - vi[i_idx],
            ])) - 4.0 * s_ij[e_idx] <= rhs_4f2
        )

    # ----- Branch flow definitions (linear in c_ii, c_ij, s_ij) -----
    # ----- + thermal limits (SOC) -----
    for k_str in cfg.branch_ids:
        k_idx = cfg.branch_to_idx[k_str]
        br = data.branches[k_str]
        fi = br['f_bus']
        ti = br['t_bus']
        f_idx = cfg.bus_to_idx[str(fi)]
        e_idx = cfg.edge_to_idx[(fi, ti)]

        g, b = calc_branch_y(br)
        tr, ti_ = calc_branch_t(br)
        g_fr = br['g_fr']
        b_fr = br['b_fr']
        g_to = br['g_to']
        b_to = br['b_to']
        tm2 = br['tap'] ** 2
        rate_a = br['rate_a']

        # From-side active power
        constraints.append(
            p_fr[k_idx] == (g + g_fr) / tm2 * c_ii[f_idx]
                          + (-g * tr + b * ti_) / tm2 * c_ij[e_idx]
                          + (-b * tr - g * ti_) / tm2 * s_ij[e_idx]
        )
        # From-side reactive power
        constraints.append(
            q_fr[k_idx] == -(b + b_fr) / tm2 * c_ii[f_idx]
                           - (-b * tr - g * ti_) / tm2 * c_ij[e_idx]
                           + (-g * tr + b * ti_) / tm2 * s_ij[e_idx]
        )
        # To-side active power (uses c_ii[f_bus] to match Julia formulation)
        constraints.append(
            p_to[k_idx] == (g + g_to) * c_ii[f_idx]
                          + (-g * tr - b * ti_) / tm2 * c_ij[e_idx]
                          + (-b * tr + g * ti_) / tm2 * s_ij[e_idx]
        )
        # To-side reactive power
        constraints.append(
            q_to[k_idx] == -(b + b_to) * c_ii[f_idx]
                           - (-b * tr + g * ti_) / tm2 * c_ij[e_idx]
                           + (-g * tr - b * ti_) / tm2 * s_ij[e_idx]
        )

        # Thermal limits: ||[p_fr, q_fr]||_2 <= rate_a  (SOC, DCP-compliant)
        constraints.append(
            cp.norm(cp.hstack([p_fr[k_idx], q_fr[k_idx]]), 2) <= rate_a
        )
        constraints.append(
            cp.norm(cp.hstack([p_to[k_idx], q_to[k_idx]]), 2) <= rate_a
        )

    # ----- Node balance (linear equalities, RHS uses pd_param/qd_param) -----
    # Build per-bus from/to branch index lists
    bus_br_fr: Dict[int, List[int]] = {i: [] for i in range(NB)}
    bus_br_to: Dict[int, List[int]] = {i: [] for i in range(NB)}
    for k_str in cfg.branch_ids:
        k_idx = cfg.branch_to_idx[k_str]
        br = data.branches[k_str]
        f_idx = cfg.bus_to_idx[str(br['f_bus'])]
        t_idx = cfg.bus_to_idx[str(br['t_bus'])]
        bus_br_fr[f_idx].append(k_idx)
        bus_br_to[t_idx].append(k_idx)

    for bi in range(NB):
        gen_idxs = cfg.bus_gen_map[bi]
        gs = cfg.bus_shunt_gs[bi]
        bs = cfg.bus_shunt_bs[bi]

        # Active balance: sum(pg) - gs*c_ii - sum(p_fr) - sum(p_to) == pd
        pg_sum = sum(pg[gi] for gi in gen_idxs) if gen_idxs else 0.0
        pfr_sum = sum(p_fr[ki] for ki in bus_br_fr[bi]) if bus_br_fr[bi] else 0.0
        pto_sum = sum(p_to[ki] for ki in bus_br_to[bi]) if bus_br_to[bi] else 0.0

        constraints.append(
            pg_sum - gs * c_ii[bi] - pfr_sum - pto_sum == pd_param[bi]
        )

        # Reactive balance: sum(qg) + bs*c_ii - sum(q_fr) - sum(q_to) == qd
        qg_sum = sum(qg[gi] for gi in gen_idxs) if gen_idxs else 0.0
        qfr_sum = sum(q_fr[ki] for ki in bus_br_fr[bi]) if bus_br_fr[bi] else 0.0
        qto_sum = sum(q_to[ki] for ki in bus_br_to[bi]) if bus_br_to[bi] else 0.0

        constraints.append(
            qg_sum + bs * c_ii[bi] - qfr_sum - qto_sum == qd_param[bi]
        )

    # ----- Learned cuts on u: A_cut @ u <= b_cut + s_cut -----
    constraints.append(A_cut @ u <= b_cut + s_cut)

    # ===== Objective =====
    # min  sum(c2*pg^2 + c1*pg + c0*u) + penalty*(slacks) + M*sum(s_cut)
    # NOTE: BIG_M kept moderate (100) so SCS (first-order solver) stays
    #       numerically well-conditioned.  The *training loss* in
    #       self_supervised.py applies a much larger penalty on s_cut
    #       to the NN, so the learned cuts still converge to feasibility.
    SLACK_PEN = 10.0   # convexity slack penalty (xi_c, xij_c, xij_s)
    CUT_PEN   = 100.0  # learned-cut slack penalty (s_cut)
    obj = (
        cp.sum(cp.multiply(cfg.cost_c2, cp.square(pg)))
        + cfg.cost_c1 @ pg
        + cfg.cost_c0 @ u
        + SLACK_PEN * (cp.sum(xi_c) + cp.sum(xij_c) + cp.sum(xij_s))
        + CUT_PEN * cp.sum(s_cut)
    )

    prob = cp.Problem(cp.Minimize(obj), constraints)

    # Verify DCP compliance
    assert prob.is_dcp(), "Problem is NOT DCP-compliant!"

    # Wrap as CvxpyLayer
    # Variables returned: u, pg, qg, vr, vi, c_ii, s_cut
    layer = CvxpyLayer(
        prob,
        parameters=[A_cut, b_cut, pd_param, qd_param],
        variables=[u, pg, qg, vr, vi, c_ii, s_cut],
    )

    return layer, cfg

# for (fi, ti) in cfg.edges:
#         e_idx = cfg.edge_to_idx[(fi, ti)]

def build_cvxpy_ac_uc_layer_supervised(
    data: MatpowerData,
    node_vr: Dict[str, float],
    node_vi: Dict[str, float],
    rho, # slack penalty
    xi_c: Dict[str, float], # indexed by [cfg.bus_ids: value]
    xij_c: Dict[Tuple[int,int], float], # indexed by [cfg.edge_to_idx[(fi, ti)] in cfg.edges : value)], I think [tuple(int, int) : value]
    xij_s: Dict[Tuple[int,int], float],
    n_cuts: int = 10,
) -> Tuple[CvxpyLayer, ACUCConfig]:
    """
    Build a CVXPY DCP-compliant convex AC-UC problem and wrap it as a
    CvxpyLayer for differentiable supervised training. Slack penalty and
    variables are taken from solved instance of Can Li. 
    

    Parameters
    ----------
    data      : parsed MATPOWER network
    node_vr   : warm-start real voltages (from rectangular MIQCQP solve)
    node_vi   : warm-start imaginary voltages
    rho       : slack penalty, taken from Can Li
    xi_c      : slack variable on bus, taken from Can Li
    xij_c     : slack variable on edge, taken from Can Li
    xij_s     : slack variable on edge, taken from Can Li
    n_cuts    : number of learned linear cuts on u (V_CONSTRAINTS)

    Returns
    -------
    layer : CvxpyLayer  with parameters [A_cut, b_cut, pd_param, qd_param]
            and variables  [u, pg, qg, vr, vi, c_ii, s_cut]
    cfg   : ACUCConfig   with index mappings and constants
    """
    cfg = _build_config(data)
    NB = cfg.n_bus
    NG = cfg.n_gen
    NK = cfg.n_branch
    NE = cfg.n_edge

    # ===== CVXPY Variables =====
    vr     = cp.Variable(NB, name="vr")
    vi     = cp.Variable(NB, name="vi")
    c_ii   = cp.Variable(NB, name="c_ii")
    c_ij   = cp.Variable(NE, name="c_ij")
    s_ij   = cp.Variable(NE, name="s_ij")
    u      = cp.Variable(NG, name="u")       # relaxed to [0,1]
    pg     = cp.Variable(NG, name="pg")
    qg     = cp.Variable(NG, name="qg")
    p_fr   = cp.Variable(NK, name="p_fr")
    q_fr   = cp.Variable(NK, name="q_fr")
    p_to   = cp.Variable(NK, name="p_to")
    q_to   = cp.Variable(NK, name="q_to")
    # xi_c   = cp.Variable(NB, name="xi_c", nonneg=True)
    # xij_c  = cp.Variable(NE, name="xij_c", nonneg=True)
    # xij_s  = cp.Variable(NE, name="xij_s", nonneg=True)
    s_cut  = cp.Variable(n_cuts, name="s_cut", nonneg=True)

    # ===== CVXPY Parameters (NN-predicted + load inputs) =====
    A_cut    = cp.Parameter((n_cuts, NG), name="A_cut")
    b_cut    = cp.Parameter(n_cuts, name="b_cut")
    pd_param = cp.Parameter(NB, name="pd")
    qd_param = cp.Parameter(NB, name="qd")

    constraints = []

    # ----- u bounds [0, 1] -----
    constraints += [u >= 0, u <= 1]

    # ----- Generator limits: pmin*u <= pg <= pmax*u, same for qg -----
    for g_idx in range(NG):
        constraints += [
            pg[g_idx] >= cfg.pmin[g_idx] * u[g_idx],
            pg[g_idx] <= cfg.pmax[g_idx] * u[g_idx],
            qg[g_idx] >= cfg.qmin[g_idx] * u[g_idx],
            qg[g_idx] <= cfg.qmax[g_idx] * u[g_idx],
        ]

    # ----- Voltage bounds: c_ii in [vmin^2, vmax^2] -----
    constraints += [
        c_ii >= cfg.vmin ** 2,
        c_ii <= cfg.vmax ** 2,
    ]

    # ----- Reference bus: vi[ref] == 0, vr[ref] >= 0 -----
    ri = cfg.ref_bus_idx
    constraints += [vi[ri] == 0, vr[ri] >= 0]

    # ----- Convex constraints 4b, 4c (diagonal W-matrix) -----
    for i_str in cfg.bus_ids:
        idx = cfg.bus_to_idx[i_str]
        vr0 = node_vr[i_str]
        vi0 = node_vi[i_str]

        # 4b: c_ii >= vr^2 + vi^2  (SOC-representable, DCP: convex <= affine)
        constraints.append(
            cp.sum_squares(cp.hstack([vr[idx], vi[idx]])) <= c_ii[idx]
        )

        # 4c: c_ii <= 2*(vr0*vr + vi0*vi) - (vr0^2 + vi0^2) + xi_c
        constraints.append(
            c_ii[idx] <= 2.0 * (vr0 * vr[idx] + vi0 * vi[idx])
                         - (vr0**2 + vi0**2)
                         + xi_c[idx]
        )

    # ----- Convex constraints 4d, 4e, 4f1, 4f2 (off-diagonal W-matrix) -----
    for (fi, ti) in cfg.edges:
        e_idx = cfg.edge_to_idx[(fi, ti)]
        i_str = str(fi)
        j_str = str(ti)
        i_idx = cfg.bus_to_idx[i_str]
        j_idx = cfg.bus_to_idx[j_str]

        vr0_i = node_vr[i_str]; vr0_j = node_vr[j_str]
        vi0_i = node_vi[i_str]; vi0_j = node_vi[j_str]

        # --- 4d ---
        # (vr_i+vr_j)^2 + (vi_i+vi_j)^2 + 4*c_ij <= rhs_4d
        rhs_4d = (
            xij_c[e_idx]
            + 2.0 * (vr[i_idx] - vr[j_idx]) * (vr0_i - vr0_j)
            + 2.0 * (vi[i_idx] - vi[j_idx]) * (vi0_i - vi0_j)
            - ((vr0_i - vr0_j)**2 + (vi0_i - vi0_j)**2)
        )
        constraints.append(
            cp.sum_squares(cp.hstack([
                vr[i_idx] + vr[j_idx],
                vi[i_idx] + vi[j_idx],
            ])) + 4.0 * c_ij[e_idx] <= rhs_4d
        )

        # --- 4e ---
        rhs_4e = (
            xij_c[e_idx]
            + 2.0 * (vr[i_idx] + vr[j_idx]) * (vr0_i + vr0_j)
            + 2.0 * (vi[i_idx] + vi[j_idx]) * (vi0_i + vi0_j)
            - ((vr0_i + vr0_j)**2 + (vi0_i + vi0_j)**2)
        )
        constraints.append(
            cp.sum_squares(cp.hstack([
                vr[i_idx] - vr[j_idx],
                vi[i_idx] - vi[j_idx],
            ])) + 4.0 * c_ij[e_idx] <= rhs_4e
        )

        # --- 4f1 ---
        rhs_4f1 = (
            xij_s[e_idx]
            + 2.0 * (vr[i_idx] + vi[j_idx]) * (vr0_i + vi0_j)
            + 2.0 * (vr[j_idx] - vi[i_idx]) * (vr0_j + vi0_i)
            - ((vr0_i + vi0_j)**2 + (vr0_j - vi0_i)**2)
        )
        constraints.append(
            cp.sum_squares(cp.hstack([
                vr[i_idx] - vi[j_idx],
                vr[j_idx] + vi[i_idx],
            ])) + 4.0 * s_ij[e_idx] <= rhs_4f1
        )

        # --- 4f2 ---
        rhs_4f2 = (
            xij_s[e_idx]
            + 2.0 * (vr[i_idx] - vi[j_idx]) * (vr0_i - vi0_j)
            + 2.0 * (vr[j_idx] + vi[i_idx]) * (vr0_j + vi0_i)
            - ((vr0_i - vi0_j)**2 + (vr0_j + vi0_i)**2)
        )
        constraints.append(
            cp.sum_squares(cp.hstack([
                vr[i_idx] + vi[j_idx],
                vr[j_idx] - vi[i_idx],
            ])) - 4.0 * s_ij[e_idx] <= rhs_4f2
        )

    # ----- Branch flow definitions (linear in c_ii, c_ij, s_ij) -----
    # ----- + thermal limits (SOC) -----
    for k_str in cfg.branch_ids:
        k_idx = cfg.branch_to_idx[k_str]
        br = data.branches[k_str]
        fi = br['f_bus']
        ti = br['t_bus']
        f_idx = cfg.bus_to_idx[str(fi)]
        e_idx = cfg.edge_to_idx[(fi, ti)]

        g, b = calc_branch_y(br)
        tr, ti_ = calc_branch_t(br)
        g_fr = br['g_fr']
        b_fr = br['b_fr']
        g_to = br['g_to']
        b_to = br['b_to']
        tm2 = br['tap'] ** 2
        rate_a = br['rate_a']

        # From-side active power
        constraints.append(
            p_fr[k_idx] == (g + g_fr) / tm2 * c_ii[f_idx]
                          + (-g * tr + b * ti_) / tm2 * c_ij[e_idx]
                          + (-b * tr - g * ti_) / tm2 * s_ij[e_idx]
        )
        # From-side reactive power
        constraints.append(
            q_fr[k_idx] == -(b + b_fr) / tm2 * c_ii[f_idx]
                           - (-b * tr - g * ti_) / tm2 * c_ij[e_idx]
                           + (-g * tr + b * ti_) / tm2 * s_ij[e_idx]
        )
        # To-side active power (uses c_ii[f_bus] to match Julia formulation)
        constraints.append(
            p_to[k_idx] == (g + g_to) * c_ii[f_idx]
                          + (-g * tr - b * ti_) / tm2 * c_ij[e_idx]
                          + (-b * tr + g * ti_) / tm2 * s_ij[e_idx]
        )
        # To-side reactive power
        constraints.append(
            q_to[k_idx] == -(b + b_to) * c_ii[f_idx]
                           - (-b * tr + g * ti_) / tm2 * c_ij[e_idx]
                           + (-g * tr - b * ti_) / tm2 * s_ij[e_idx]
        )

        # Thermal limits: ||[p_fr, q_fr]||_2 <= rate_a  (SOC, DCP-compliant)
        constraints.append(
            cp.norm(cp.hstack([p_fr[k_idx], q_fr[k_idx]]), 2) <= rate_a
        )
        constraints.append(
            cp.norm(cp.hstack([p_to[k_idx], q_to[k_idx]]), 2) <= rate_a
        )

    # ----- Node balance (linear equalities, RHS uses pd_param/qd_param) -----
    # Build per-bus from/to branch index lists
    bus_br_fr: Dict[int, List[int]] = {i: [] for i in range(NB)}
    bus_br_to: Dict[int, List[int]] = {i: [] for i in range(NB)}
    for k_str in cfg.branch_ids:
        k_idx = cfg.branch_to_idx[k_str]
        br = data.branches[k_str]
        f_idx = cfg.bus_to_idx[str(br['f_bus'])]
        t_idx = cfg.bus_to_idx[str(br['t_bus'])]
        bus_br_fr[f_idx].append(k_idx)
        bus_br_to[t_idx].append(k_idx)

    for bi in range(NB):
        gen_idxs = cfg.bus_gen_map[bi]
        gs = cfg.bus_shunt_gs[bi]
        bs = cfg.bus_shunt_bs[bi]

        # Active balance: sum(pg) - gs*c_ii - sum(p_fr) - sum(p_to) == pd
        pg_sum = sum(pg[gi] for gi in gen_idxs) if gen_idxs else 0.0
        pfr_sum = sum(p_fr[ki] for ki in bus_br_fr[bi]) if bus_br_fr[bi] else 0.0
        pto_sum = sum(p_to[ki] for ki in bus_br_to[bi]) if bus_br_to[bi] else 0.0

        constraints.append(
            pg_sum - gs * c_ii[bi] - pfr_sum - pto_sum == pd_param[bi]
        )

        # Reactive balance: sum(qg) + bs*c_ii - sum(q_fr) - sum(q_to) == qd
        qg_sum = sum(qg[gi] for gi in gen_idxs) if gen_idxs else 0.0
        qfr_sum = sum(q_fr[ki] for ki in bus_br_fr[bi]) if bus_br_fr[bi] else 0.0
        qto_sum = sum(q_to[ki] for ki in bus_br_to[bi]) if bus_br_to[bi] else 0.0

        constraints.append(
            qg_sum + bs * c_ii[bi] - qfr_sum - qto_sum == qd_param[bi]
        )

    # ----- Learned cuts on u: A_cut @ u <= b_cut + s_cut -----
    constraints.append(A_cut @ u <= b_cut + s_cut)

    # ===== Objective =====
    # min  sum(c2*pg^2 + c1*pg + c0*u) + penalty*(slacks) + M*sum(s_cut)
    # NOTE: BIG_M kept moderate (100) so SCS (first-order solver) stays
    #       numerically well-conditioned.  The *training loss* in
    #       self_supervised.py applies a much larger penalty on s_cut
    #       to the NN, so the learned cuts still converge to feasibility.
    SLACK_PEN = rho   # convexity slack penalty (xi_c, xij_c, xij_s)
    CUT_PEN   = 100.0  # learned-cut slack penalty (s_cut)
    obj = (
        cp.sum(cp.multiply(cfg.cost_c2, cp.square(pg)))
        + cfg.cost_c1 @ pg
        + cfg.cost_c0 @ u
        + SLACK_PEN * (cp.sum(xi_c) + cp.sum(xij_c) + cp.sum(xij_s))
        + CUT_PEN * cp.sum(s_cut)
    )

    prob = cp.Problem(cp.Minimize(obj), constraints)

    # Verify DCP compliance
    assert prob.is_dcp(), "Problem is NOT DCP-compliant!"

    # Wrap as CvxpyLayer
    # Variables returned: u, pg, qg, vr, vi, c_ii, s_cut
    layer = CvxpyLayer(
        prob,
        parameters=[A_cut, b_cut, pd_param, qd_param],
        variables=[u, pg, qg, vr, vi, c_ii, s_cut],
    )

    return layer, cfg
