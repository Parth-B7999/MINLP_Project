"""
data_utils.py
=============
Parses MATPOWER .m case files and provides branch admittance helpers.

Mirrors the Julia structs and functions from formulation.jl:
    struct MatpowerData
    _parse_file_data(file_path)
    _get_edges(data)
    PowerModels.calc_branch_y(branch)
    PowerModels.calc_branch_t(branch)
    get_old_voltages(model, data)          <- gurobipy model variant

All power quantities are stored in per-unit (base: mpc.baseMVA, typically 100 MVA).
Cost coefficients are scaled so the objective f(pg_pu) = cost[0]*pg_pu^2
+ cost[1]*pg_pu + cost[2]*u is dimensionally consistent with per-unit pg.
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class MatpowerData:
    """
    Mirrors Julia's MatpowerData struct.

    All dicts are keyed by *string* IDs (e.g. '1', '2', ...).
    Edge tuples (f_bus, t_bus) use *int* IDs, matching Julia's convention
    where c_ij / s_ij are indexed by (Int, Int) tuples.
    """
    buses:    Dict[str, Dict[str, Any]]   # bus_id (str) -> bus data
    gens:     Dict[str, Dict[str, Any]]   # gen_id (str, 1-indexed) -> gen data
    branches: Dict[str, Dict[str, Any]]   # branch_id (str, 1-indexed) -> branch data
    loads:    Dict[str, Dict[str, Any]]   # load_id (str) -> load data
    shunts:   Dict[str, Dict[str, Any]]   # bus_id (str) -> shunt data


# ---------------------------------------------------------------------------
# Raw MATPOWER .m parser
# ---------------------------------------------------------------------------

def _parse_matpower_m(file_path: str) -> dict:
    """
    Parse raw MATPOWER v2 .m case file into numpy arrays.

    Handles the standard script format:
        mpc.baseMVA = 100;
        mpc.bus     = [...];
        mpc.gen     = [...];
        mpc.branch  = [...];
        mpc.gencost = [...];
    """
    with open(file_path, 'r') as fh:
        content = fh.read()

    def extract_matrix(name: str) -> Optional[np.ndarray]:
        pattern = rf'mpc\.{name}\s*=\s*\[(.*?)\]\s*;'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return None
        matrix_str = match.group(1)
        matrix_str = re.sub(r'%[^\n]*', '', matrix_str)   # strip % comments
        rows = []
        for line in matrix_str.split('\n'):
            line = line.strip().rstrip(';').strip()
            if line:
                vals = line.split()
                if vals:
                    try:
                        rows.append([float(v) for v in vals])
                    except ValueError:
                        pass
        return np.array(rows) if rows else None

    basemva_match = re.search(r'mpc\.baseMVA\s*=\s*([\d.]+)', content)
    base_mva = float(basemva_match.group(1)) if basemva_match else 100.0

    return {
        'baseMVA': base_mva,
        'bus':     extract_matrix('bus'),
        'gen':     extract_matrix('gen'),
        'branch':  extract_matrix('branch'),
        'gencost': extract_matrix('gencost'),
    }


# ---------------------------------------------------------------------------
# Full parse + standardize (mirrors PowerModels pipeline)
# ---------------------------------------------------------------------------

def parse_file_data(file_path: str) -> MatpowerData:
    """
    Parse a MATPOWER .m case file and return a MatpowerData object.

    Mirrors the Julia pipeline:
        data = PowerModels.parse_file(file_path)
        PowerModels.standardize_cost_terms!(data, order=2)
        PowerModels.calc_thermal_limits!(data)

    Per-unit conversion
    -------------------
    MATPOWER stores power in MW / MVAr and cost in $/hr.
    After parsing all power quantities are divided by baseMVA so that
    pg, qg, pd, qd, rate_a are all in per-unit.

    Cost scaling (standardize_cost_terms! behaviour)
    -------------------------------------------------
    MATPOWER polynomial cost with pg in MW:
        f(pg_MW) = c2*pg_MW^2 + c1*pg_MW + c0
    Rewritten for per-unit pg (pg_pu = pg_MW / baseMVA):
        f(pg_pu) = c2*baseMVA^2 * pg_pu^2 + c1*baseMVA * pg_pu + c0
    So we store: cost = [c2*baseMVA^2, c1*baseMVA, c0]
    """
    raw      = _parse_matpower_m(file_path)
    base_mva = raw['baseMVA']
    bus_arr      = raw['bus']
    gen_arr      = raw['gen']
    branch_arr   = raw['branch']
    gencost_arr  = raw['gencost']

    # ------------------------------------------------------------------
    # Buses
    # MATPOWER bus columns (0-indexed):
    #   0=BUS_I  1=TYPE  2=PD  3=QD  4=GS  5=BS  6=AREA
    #   7=VM  8=VA  9=BASEKV  10=ZONE  11=VMAX  12=VMIN
    # ------------------------------------------------------------------
    buses: Dict[str, Dict[str, Any]] = {}
    for row in bus_arr:
        bid = str(int(row[0]))
        buses[bid] = {
            'bus_i':    int(row[0]),
            'bus_type': int(row[1]),    # 1=PQ, 2=PV, 3=ref/slack
            'gs':       row[4] / base_mva,   # shunt conductance (pu)
            'bs':       row[5] / base_mva,   # shunt susceptance (pu)
            'vmin':     row[12],
            'vmax':     row[11],
            'vm':       row[7],
            'va':       float(np.deg2rad(row[8])),
        }

    # ------------------------------------------------------------------
    # Generator costs  (standardize_cost_terms! -> order=2 polynomial)
    # MATPOWER gencost columns (0-indexed):
    #   0=MODEL  1=STARTUP  2=SHUTDOWN  3=NCOST  4..=PARAMS
    # For MODEL=2 (polynomial), PARAMS = [c_n, ..., c1, c0]
    # ------------------------------------------------------------------
    cost_lookup: Dict[str, List[float]] = {}
    if gencost_arr is not None:
        for idx, row in enumerate(gencost_arr):
            gid       = str(idx + 1)
            model_typ = int(row[0])
            ncost     = int(row[3])
            coeffs    = row[4: 4 + ncost].tolist()
            if model_typ == 2:
                if ncost >= 3:
                    c2, c1, c0 = coeffs[0], coeffs[1], coeffs[2]
                elif ncost == 2:
                    c2, c1, c0 = 0.0, coeffs[0], coeffs[1]
                elif ncost == 1:
                    c2, c1, c0 = 0.0, 0.0, coeffs[0]
                else:
                    c2, c1, c0 = 0.0, 0.0, 0.0
                cost_lookup[gid] = [
                    c2 * base_mva ** 2,   # $/hr / pu^2
                    c1 * base_mva,        # $/hr / pu
                    c0,                   # $/hr  (no-load cost when u=1)
                ]
            else:
                # Piecewise-linear: zero cost approximation (placeholder)
                cost_lookup[gid] = [0.0, 0.0, 0.0]

    # ------------------------------------------------------------------
    # Generators
    # MATPOWER gen columns (0-indexed):
    #   0=BUS  1=PG  2=QG  3=QMAX  4=QMIN  5=VG  6=MBASE
    #   7=STATUS  8=PMAX  9=PMIN
    # ------------------------------------------------------------------
    gens: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(gen_arr):
        gid = str(idx + 1)
        gens[gid] = {
            'gen_bus':    str(int(row[0])),
            'pg':         row[1]  / base_mva,
            'qg':         row[2]  / base_mva,
            'qmax':       row[3]  / base_mva,
            'qmin':       row[4]  / base_mva,
            'pmax':       row[8]  / base_mva,
            'pmin':       row[9]  / base_mva,
            'gen_status': int(row[7]),
            # cost = [c2_pu, c1_pu, c0]  consistent with per-unit pg
            'cost':       cost_lookup.get(gid, [0.0, 0.0, 0.0]),
        }

    # ------------------------------------------------------------------
    # Branches
    # MATPOWER branch columns (0-indexed):
    #   0=F_BUS  1=T_BUS  2=BR_R  3=BR_X  4=BR_B  5=RATE_A
    #   6=RATE_B  7=RATE_C  8=TAP  9=SHIFT  10=STATUS
    #   11=ANGMIN  12=ANGMAX
    #
    # calc_thermal_limits!: RATE_A=0 means unlimited -> use 9999 MVA.
    # Line charging b_c is split 50/50 as b_fr and b_to (standard Pi model).
    # ------------------------------------------------------------------
    branches: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(branch_arr):
        bid   = str(idx + 1)
        tap   = row[8] if row[8] != 0.0 else 1.0
        shift = float(np.deg2rad(row[9]))
        b_c = row[4]   # total line-charging susceptance (pu)
        if row[5] > 0.0:
            rate_a_pu = row[5] / base_mva
        else:
            # MATPOWER rate_a=0 means "unlimited" — compute from pi-model
            # admittance (mirrors PowerModels calc_thermal_limits!)
            br_r, br_x = row[2], row[3]
            tap_val = row[8] if row[8] != 0.0 else 1.0
            z2 = br_r**2 + br_x**2
            if z2 > 0.0:
                g_s = br_r / z2
                b_s = -br_x / z2
                rate_a_pu = float(np.sqrt(g_s**2 + (b_s + b_c / 2.0)**2) / (tap_val**2))
            else:
                rate_a_pu = 99.99  # fallback (lossless line)
        branches[bid] = {
            'f_bus':     int(row[0]),
            't_bus':     int(row[1]),
            'br_r':      row[2],
            'br_x':      row[3],
            'br_b':      b_c,
            'rate_a':    rate_a_pu,
            'tap':       tap,
            'shift':     shift,
            'g_fr':      0.0,           # from-side shunt conductance (Pi model)
            'b_fr':      b_c / 2.0,    # from-side shunt susceptance
            'g_to':      0.0,           # to-side shunt conductance
            'b_to':      b_c / 2.0,    # to-side shunt susceptance
            'br_status': int(row[10]),
            'angmin':    float(np.deg2rad(row[11])) if len(row) > 11 else -np.pi,
            'angmax':    float(np.deg2rad(row[12])) if len(row) > 12 else  np.pi,
        }

    # ------------------------------------------------------------------
    # Loads  (PowerModels separates loads from bus data)
    # In MATPOWER, PD / QD sit in the bus array (columns 2, 3).
    # ------------------------------------------------------------------
    loads: Dict[str, Dict[str, Any]] = {}
    load_idx = 1
    for row in bus_arr:
        bid = str(int(row[0]))
        pd  = row[2] / base_mva
        qd  = row[3] / base_mva
        if abs(pd) > 1e-10 or abs(qd) > 1e-10:
            loads[str(load_idx)] = {
                'load_bus': bid,
                'pd':       pd,
                'qd':       qd,
                'status':   1,
            }
            load_idx += 1

    # ------------------------------------------------------------------
    # Shunts  (GS / BS per bus, columns 4 & 5 in bus array)
    # Already converted to pu above when reading buses.
    # ------------------------------------------------------------------
    shunts: Dict[str, Dict[str, Any]] = {}
    for row in bus_arr:
        bid = str(int(row[0]))
        gs  = row[4] / base_mva
        bs  = row[5] / base_mva
        if abs(gs) > 1e-10 or abs(bs) > 1e-10:
            shunts[bid] = {
                'shunt_bus': bid,
                'gs':        gs,
                'bs':        bs,
                'status':    1,
            }

    return MatpowerData(
        buses=buses, gens=gens, branches=branches,
        loads=loads, shunts=shunts,
    )


# ---------------------------------------------------------------------------
# Branch helpers  (mirror PowerModels.calc_branch_y / calc_branch_t)
# ---------------------------------------------------------------------------

def get_edges(data: MatpowerData) -> List[Tuple[int, int]]:
    """
    Return list of (f_bus_int, t_bus_int) for every branch.
    Mirrors Julia's _get_edges(data::MatpowerData).
    """
    return [(br['f_bus'], br['t_bus']) for br in data.branches.values()]


def calc_branch_y(branch: Dict[str, Any]) -> Tuple[float, float]:
    """
    Compute series admittance (g, b) from branch impedance.
    Mirrors PowerModels.calc_branch_y(branch).
    """
    y = 1.0 / complex(branch['br_r'], branch['br_x'])
    return y.real, y.imag


def calc_branch_t(branch: Dict[str, Any]) -> Tuple[float, float]:
    """
    Compute complex tap components (tr, ti) from tap magnitude and shift angle.
    Mirrors PowerModels.calc_branch_t(branch).
    """
    tr = branch['tap'] * np.cos(branch['shift'])
    ti = branch['tap'] * np.sin(branch['shift'])
    return tr, ti


# ---------------------------------------------------------------------------
# Voltage extraction from a solved gurobipy model
# (mirrors get_old_voltages in Julia)
# ---------------------------------------------------------------------------

def get_old_voltages(
    model,
    data: MatpowerData,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extract warm-start voltages from a solved gurobipy rectangular model.

    Returns
    -------
    node_vr : dict  bus_id (str) -> real voltage value at t=1
    node_vi : dict  bus_id (str) -> imaginary voltage value at t=1

    Mirrors Julia's get_old_voltages(model, data).
    """
    node_vr: Dict[str, float] = {}
    node_vi: Dict[str, float] = {}
    for bus_id in data.buses:
        node_vr[bus_id] = model._vr[bus_id, 1].X
        node_vi[bus_id] = model._vi[bus_id, 1].X
    return node_vr, node_vi
