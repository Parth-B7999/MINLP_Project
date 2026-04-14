import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np

def build_diffopt_qcac_layer(data, num_cuts=10):
    """
    Builds the differentiable CVXPY layer for the QCAC AC-UC relaxation.
    All non-linearities are translated to strict DCP-compliant functions.
    """
    print("Building Differentiable CVXPY Layer (This may take a moment)...")
    
    # -----------------------------------------------------------------
    # 0. Extract Data and Create Index Mappings
    # (CVXPY uses 0-indexed arrays, so we must map string IDs to integers)
    # -----------------------------------------------------------------
    bus_ids = list(data.buses.keys())
    gen_ids = list(data.gens.keys())
    branch_ids = list(data.branches.keys())
    
    b_idx = {b_id: i for i, b_id in enumerate(bus_ids)}
    g_idx = {g_id: i for i, g_id in enumerate(gen_ids)}
    br_idx = {br_id: i for i, br_id in enumerate(branch_ids)}
    
    N_b = len(bus_ids)
    N_g = len(gen_ids)
    N_br = len(branch_ids)
    
    # -----------------------------------------------------------------
    # 1. Variables (Continuous Decisions)
    # -----------------------------------------------------------------
    # Voltages
    vr = cp.Variable(N_b)
    vi = cp.Variable(N_b)
    c_ii = cp.Variable(N_b)
    c_ij = cp.Variable(N_br)
    s_ij = cp.Variable(N_br)
    
    # Generation & Commitment (Relaxed to continuous [0,1])
    u_relax = cp.Variable(N_g)
    pg = cp.Variable(N_g)
    qg = cp.Variable(N_g)
    
    # Branch Flows
    p_fr = cp.Variable(N_br)
    q_fr = cp.Variable(N_br)
    p_to = cp.Variable(N_br)
    q_to = cp.Variable(N_br)
    
    # Slack Variables (Must be non-negative)
    xi_c = cp.Variable(N_b, nonneg=True)
    xij_c = cp.Variable(N_br, nonneg=True)
    xij_s = cp.Variable(N_br, nonneg=True)
    
    # -----------------------------------------------------------------
    # 2. Parameters (NN Outputs + Environment Inputs)
    # -----------------------------------------------------------------
    # NN Outputs (Gradients will flow back through these!)
    vr_base = cp.Parameter(N_b)
    vi_base = cp.Parameter(N_b)
    rho = cp.Parameter(nonneg=True)
    A_cut = cp.Parameter((num_cuts, N_g))
    b_cut = cp.Parameter(num_cuts)
    
    # 1. Define the slack variable
    s_cut = cp.Variable(num_cuts, nonneg=True)

    # Environment (Load Profiles)
    pd = cp.Parameter(N_b)
    qd = cp.Parameter(N_b)
    
    constraints = []
    
    # --- FIX: Explicit Continuous Relaxation Bounds for u ---
    constraints.append(u_relax >= 0.0)
    constraints.append(u_relax <= 1.0)
    # -----------------------------------------------------------------
    # 3. Reference Bus & Generator Bounds
    # -----------------------------------------------------------------
    ref_buses = [b_idx[k] for k, v in data.buses.items() if v['bus_type'] == 3]
    for i in ref_buses:
        constraints.append(vi[i] == 0.0)
        constraints.append(vr[i] >= 0.0)
        
    for g_id, gen in data.gens.items():
        g = g_idx[g_id]
        constraints.append(pg[g] >= gen['pmin'] * u_relax[g])
        constraints.append(pg[g] <= gen['pmax'] * u_relax[g])
        constraints.append(qg[g] >= gen['qmin'] * u_relax[g])
        constraints.append(qg[g] <= gen['qmax'] * u_relax[g])

    # -----------------------------------------------------------------
    # 4. Nodal QCAC Constraints (Eq 4b, 4c)
    # -----------------------------------------------------------------
    for b_id, bus in data.buses.items():
        i = b_idx[b_id]
        
        # Voltage Magnitude Bounds
        constraints.append(c_ii[i] >= bus['vmin']**2)
        constraints.append(c_ii[i] <= bus['vmax']**2)
        
        # 4b: Strict convexity (SOCP representable)
        constraints.append(c_ii[i] >= cp.square(vr[i]) + cp.square(vi[i]))
        
        # 4c: Taylor Expansion RHS (Uses cp.multiply for Param * Var)
        lin_term = 2.0 * (cp.multiply(vr_base[i], vr[i]) + cp.multiply(vi_base[i], vi[i]))
        const_term = cp.square(vr_base[i]) + cp.square(vi_base[i])
        constraints.append(c_ii[i] <= lin_term - const_term + xi_c[i])

    # -----------------------------------------------------------------
    # 5. Branch QCAC Constraints (Eq 4d, 4e, 4f) & Physics
    # -----------------------------------------------------------------
    from .data_utils import calc_branch_y, calc_branch_t # Import helpers
    
    for br_id, branch in data.branches.items():
        k = br_idx[br_id]
        i = b_idx[str(branch['f_bus'])]
        j = b_idx[str(branch['t_bus'])]
        
        # --- Physics Setup ---
        g, b = calc_branch_y(branch)
        tr, ti_ = calc_branch_t(branch)
        g_fr, b_fr = branch['g_fr'], branch['b_fr']
        g_to, b_to = branch['g_to'], branch['b_to']
        tm2 = branch['tap'] ** 2
        rate_a = branch['rate_a']
        
        # --- 4d ---
        rhs_4d = (xij_c[k] 
                  + 2.0 * cp.multiply(vr[i] - vr[j], vr_base[i] - vr_base[j])
                  + 2.0 * cp.multiply(vi[i] - vi[j], vi_base[i] - vi_base[j])
                  - (cp.square(vr_base[i] - vr_base[j]) + cp.square(vi_base[i] - vi_base[j])))
        constraints.append(cp.square(vr[i] + vr[j]) + cp.square(vi[i] + vi[j]) - 4.0 * c_ij[k] <= rhs_4d)
        
        # --- 4e ---
        rhs_4e = (xij_c[k] 
                  + 2.0 * cp.multiply(vr[i] + vr[j], vr_base[i] + vr_base[j])
                  + 2.0 * cp.multiply(vi[i] + vi[j], vi_base[i] + vi_base[j])
                  - (cp.square(vr_base[i] + vr_base[j]) + cp.square(vi_base[i] + vi_base[j])))
        constraints.append(cp.square(vr[i] - vr[j]) + cp.square(vi[i] - vi[j]) + 4.0 * c_ij[k] <= rhs_4e)
        
        # --- 4f1 (Fixed Sign!) ---
        rhs_4f1 = (xij_s[k] 
                   + 2.0 * cp.multiply(vr[i] + vi[j], vr_base[i] + vi_base[j])
                   + 2.0 * cp.multiply(vr[j] - vi[i], vr_base[j] - vi_base[i])
                   - (cp.square(vr_base[i] + vi_base[j]) + cp.square(vr_base[j] - vi_base[i])))
        constraints.append(cp.square(vr[i] - vi[j]) + cp.square(vr[j] + vi[i]) + 4.0 * s_ij[k] <= rhs_4f1)
        
        # --- 4f2 ---
        rhs_4f2 = (xij_s[k] 
                   + 2.0 * cp.multiply(vr[i] - vi[j], vr_base[i] - vi_base[j])
                   + 2.0 * cp.multiply(vr[j] + vi[i], vr_base[j] + vi_base[i])
                   - (cp.square(vr_base[i] - vi_base[j]) + cp.square(vr_base[j] + vi_base[i])))
        constraints.append(cp.square(vr[i] + vi[j]) + cp.square(vr[j] - vi[i]) - 4.0 * s_ij[k] <= rhs_4f2)

        # --- Active/Reactive Power Flow (Linear in W-matrix) ---
        constraints.append(p_fr[k] == (g + g_fr)/tm2 * c_ii[i] + (-g*tr + b*ti_)/tm2 * c_ij[k] + (-b*tr - g*ti_)/tm2 * s_ij[k])
        constraints.append(q_fr[k] == -(b + b_fr)/tm2 * c_ii[i] - (-b*tr - g*ti_)/tm2 * c_ij[k] + (-g*tr + b*ti_)/tm2 * s_ij[k])
        
        # NOTE: c_jj fix applied below
        constraints.append(p_to[k] == (g + g_to) * c_ii[j] + (-g*tr - b*ti_)/tm2 * c_ij[k] + (-b*tr + g*ti_)/tm2 * s_ij[k])
        constraints.append(q_to[k] == -(b + b_to) * c_ii[j] - (-b*tr + g*ti_)/tm2 * c_ij[k] + (-g*tr - b*ti_)/tm2 * s_ij[k])
        
        # Thermal Limits
        constraints.append(cp.square(p_fr[k]) + cp.square(q_fr[k]) <= rate_a**2)
        constraints.append(cp.square(p_to[k]) + cp.square(q_to[k]) <= rate_a**2)

    # -----------------------------------------------------------------
    # 6. Nodal Power Balance (Kirchhoff)
    # -----------------------------------------------------------------
    for b_id, bus in data.buses.items():
        i = b_idx[b_id]
        
        # Find attached generators, shunts, and branches
        bus_gens = [g_idx[g] for g, d in data.gens.items() if d['gen_bus'] == b_id]
        br_fr = [br_idx[k] for k, b in data.branches.items() if str(b['f_bus']) == b_id]
        br_to = [br_idx[k] for k, b in data.branches.items() if str(b['t_bus']) == b_id]
        gs = sum(s['gs'] for s in data.shunts.values() if s['shunt_bus'] == b_id)
        bs = sum(s['bs'] for s in data.shunts.values() if s['shunt_bus'] == b_id)
        
        # P Balance
        pg_sum = cp.sum(pg[bus_gens]) if bus_gens else 0.0
        pfr_sum = cp.sum(p_fr[br_fr]) if br_fr else 0.0
        pto_sum = cp.sum(p_to[br_to]) if br_to else 0.0
        constraints.append(pg_sum - gs * c_ii[i] - pfr_sum - pto_sum == pd[i])
        
        # Q Balance
        qg_sum = cp.sum(qg[bus_gens]) if bus_gens else 0.0
        qfr_sum = cp.sum(q_fr[br_fr]) if br_fr else 0.0
        qto_sum = cp.sum(q_to[br_to]) if br_to else 0.0
        constraints.append(qg_sum + bs * c_ii[i] - qfr_sum - qto_sum == qd[i])

    # -----------------------------------------------------------------
    # 7. Learned Cuts (Semi-Supervised Integer Tightening)
    # -----------------------------------------------------------------
    # constraints.append(A_cut @ u_relax <= b_cut)
    # 2. Add the slack variable to the constraint so it is always feasible
    constraints.append(A_cut @ u_relax <= b_cut + s_cut)

    # -----------------------------------------------------------------
    # 8. Objective Function
    # -----------------------------------------------------------------
    cost_expr = 0
    for g_id, gen in data.gens.items():
        g = g_idx[g_id]
        c2, c1, c0 = gen['cost']
        cost_expr += c2 * cp.square(pg[g]) + c1 * pg[g] + c0 * u_relax[g]
        
    slack_sum = cp.sum(xi_c) + cp.sum(xij_c) + cp.sum(xij_s)
    
    # Minimize generation cost PLUS the dynamically learned slack penalty
    obj = cp.Minimize(cost_expr + rho * slack_sum)
    
    # -----------------------------------------------------------------
    # 9. Compile the Layer
    # -----------------------------------------------------------------
    prob = cp.Problem(obj, constraints)
    
    if not prob.is_dcp():
        raise ValueError("CVXPY formulation is not DCP! Check constraint syntax.")
        
    print("Layer compilation complete. DCP Compliant: TRUE")
    
    # Define EXACTLY which parameters PyTorch is allowed to inject and backpropagate through
    # Define EXACTLY which variables we want PyTorch to extract after the forward pass
    layer = CvxpyLayer(prob, 
                       parameters=[vr_base, vi_base, rho, A_cut, b_cut, pd, qd], 
                       variables=[u_relax, pg, qg, vr, vi, s_cut, xi_c, xij_c, xij_s])
    
    return layer, b_idx, g_idx