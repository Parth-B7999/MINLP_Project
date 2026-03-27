module formulation

# Optimization Problem Structure
using JuMP, DiffOpt, Ipopt, Gurobi

using PowerModels

export convex_ac_uc, build_convex_ac_uc, build_diff_opf, ac_uc, mp_ac_uc
"""
Here is where we will maintain the problem structure and the core functions to running this problem.
"""

struct MatpowerData
    buses::Dict{String, Any}
    gens::Dict{String, Any}
    branches::Dict{String, Any}
    loads::Dict{String, Any}
    shunts::Dict{String, Any}
end

function MatpowerData(data::Dict{String, Any})
    return MatpowerData(
        data["bus"],
        data["gen"],
        data["branch"],
        data["load"],
        get(data, "shunt", Dict{String, Any}()) # Default to empty Dict
    )
end


function ac_uc(file_path::String, type="Rectangular")
    if type == "Rectangular"
        return build_single_ac_uc_rectangular(file_path)
    else
        return build_single_period_ac_uc_polar(file_path)
    end
end

function mp_ac_uc(file_path::String, demand_curve::Vector{Float64}, type="Rectangular")
    if type == "Rectangular"
        return build_mp_ac_uc_rectangular(file_path, demand_curve)
    else
        return build_mp_ac_uc_polar(file_path, demand_curve)
    end
end

function convex_ac_uc(file_path::String, params)
    """
    * file_path: string going to the .m file for the IEEE library
    * params: Dict that contains the parameters that we are training on: loads and base voltages
    """
    # check the 'params' parameter to ensure that it fites for the function build_convex_ac_uc()
    # 1. 


    return build_convex_ac_uc(file_path, params["node_vr"], params["node_vi"])
end

function build_single_period_ac_uc_polar(file_path::String)    
    data = _parse_file_data(file_path)
    T = [1]
    # Initialize the JuMP Model
    model = Model(Gurobi.Optimizer)

    # Add variables 
    _add_acuc_var_polar!(model, data, T)

    # Add objective
    _add_mincost_obj!(model, data, T)

    # Add constraints
    _add_ref_limits_polar!(model, data, T)

    _add_gen_limits!(model, data, T)

    _add_polar_branchflow!(model, data, T)

    _add_node_bal_polar!(model, data, T)

    return model
end

function build_mp_ac_uc_polar(file_path::String, demand_curve::Vector{Float64})
    """
    demand_curve::Vector{Float64} - scaled demand levels for each T
    akin to ExaModelsPower
    """
    data = _parse_file_data(file_path)
    T = [i for (i,val) in enumerate(demand_curve)]

    # Initialize the JuMP Model
    model = Model(Gurobi.Optimizer)

    # Add variables 
    _add_acuc_var_polar!(model, data, T)

    # Add objective
    _add_mincost_obj!(model, data, T)

    # Add constraints
    _add_ref_limits_polar!(model, data, T)

    _add_gen_limits!(model, data, T)

    _add_polar_branchflow!(model, data, T)

    _add_node_bal_polar!(model, data, demand_curve)

    return model
end

function build_mp_ac_uc_rectangular(file_path::String, demand_curve::Vector{Float64})
    """
    demand_curve::Vector{Float64} - scaled demand levels for each T
    akin to ExaModelsPower
    """
    data = _parse_file_data(file_path)
    T = [i for (i, _) in enumerate(demand_curve)]

    # Initialize the JuMP Model
    model = Model(Gurobi.Optimizer)

    # Add variables 
    _add_acuc_var_rectangular!(model, data, T)

    # Add objective
    _add_mincost_obj!(model, data, T)

    # Add constraints

    _add_ref_limits_rectangular!(model, data, T)
    _add_gen_limits!(model, data, T)

    _add_rectangular_branchflow!(model, data, T)

    _add_node_bal_rectangular!(model, data, demand_curve)

    return model
end

function build_single_ac_uc_rectangular(file_path::String)
    data = _parse_file_data(file_path)
    T = [1]

    # Initialize the JuMP Model
    model = Model(Gurobi.Optimizer)

    # Add variables 
    _add_acuc_var_rectangular!(model, data, T)

    # Add objective
    _add_mincost_obj!(model, data, T)
    # Add constraints
    _add_ref_limits_rectangular!(model, data, T) 
    _add_gen_limits!(model, data, T)
    _add_rectangular_branchflow!(model, data, T)
    _add_node_bal_rectangular!(model, data, T)

    return model

end

function build_convex_ac_uc(file_path::String, node_vr, node_vi, node_pd=nothing, node_qd=nothing)
    """
    Using Constante-Flores and Li 2026 convex QCQP approximation of ACPF
    The reformulation comes from: 
    c_ii = vr^2 + vi^2 forall n (1l)
    c_ij = vi_i vi_j + vr_i vr_j forall i,j in E (1m)
    s_ij = vr_i vi_j - vr_j vi_i forall i,j in E (1n)

    * node_vr - dictionary that contains previous real voltage information for each bus
    * node_vi - dictionary that contains previous imaginary voltage information for each bus
    * node_pd - dictionary that contains the active power demand for each node
    * node_qd - dictionary that contains the reactive power demand for each node
    """
    data = _parse_file_data(file_path)
    T = [1]

    # Initialize the JuMP Model
    model = Model(Gurobi.Optimizer)

    # Add variables 
    _add_acuc_var_rectangular!(model, data, T, convex=true)
    _add_convex_constraints!(model, data, T, node_vr, node_vi)

    # Add objective
    _add_mincost_obj!(model, data, T, convex=true)
    # Add constraints
    _add_ref_limits_rectangular!(model, data, T) 
    _add_gen_limits!(model, data, T)
    _add_rectangular_branchflow!(model, data, T)
    param_loads = Dict("pd" => node_pd, "qd" => node_qd)
    _add_node_bal_rectangular!(model, data, T)

    return model
end

"""
    build_diff_opf(file_path, θ̂_vr, θ̂_vi) -> JuMP.Model

Differentiable version of `build_convex_ac_uc` for use as an OPF layer inside
a neural network. The linearization point (node_vr, node_vi) is declared as
DiffOpt `Parameter` variables so that `reverse_differentiate!` can return
dL/d(node_vr) and dL/d(node_vi) after a backward pass.

The unit commitment variables `u` are relaxed to [0,1] because DiffOpt
requires a continuous, convex problem to form KKT conditions.

# Arguments
- `file_path`: path to the Matpower `.m` case file
- `θ̂_vr`: predicted real voltage linearization point — Vector of length n_buses,
           ordered by ascending numeric bus ID (matches `encode_u` convention)
- `θ̂_vi`: predicted imaginary voltage linearization point — same ordering

# Usage (in rrule)
```julia
model = build_diff_opf(file, θ̂_vr, θ̂_vi)
optimize!(model)
# seed dL/dx*, then:
DiffOpt.reverse_differentiate!(model)
dθ̂_vr = DiffOpt.get_reverse_parameter.(model, model[:node_vr])
dθ̂_vi = DiffOpt.get_reverse_parameter.(model, model[:node_vi])
```
"""
function build_diff_opf(file_path::String, θ̂_vr::Vector{Float64}, θ̂_vi::Vector{Float64})
    data = _parse_file_data(file_path)
    T = [1]

    model = DiffOpt.diff_model(Gurobi.Optimizer)

    # Bus IDs sorted ascending — must match the ordering used in encode_u
    bus_ids = sort(collect(keys(data.buses)), by = k -> parse(Int, k))
    @assert length(θ̂_vr) == length(bus_ids) && length(θ̂_vi) == length(bus_ids)

    # Declare linearization point as DiffOpt Parameters.
    # These become the cut coefficients A(u, θ̂) x ≤ B(u, θ̂) in _add_convex_constraints!
    # DiffOpt tracks them through the KKT system so reverse_differentiate!
    # can return dL/dθ̂ for the FFNN backward pass.
    @variable(model, node_vr[bus_ids] in Parameter.(θ̂_vr))
    @variable(model, node_vi[bus_ids] in Parameter.(θ̂_vi))

    # UC decisions excluded — the OPF layer is a pure continuous convex QCQP.
    # UC is a MIQCQP concern; the surrogate learns to approximate its solution
    # via the linearization point θ̂ without re-introducing integer variables.
    _add_acuc_var_rectangular!(model, data, T, convex=true, include_uc=false)
    _add_convex_constraints!(model, data, T, node_vr, node_vi)

    _add_mincost_obj!(model, data, T, convex=true, no_uc=true)
    _add_ref_limits_rectangular!(model, data, T)
    _add_continuous_gen_limits!(model, data, T)
    _add_rectangular_branchflow!(model, data, T)
    _add_node_bal_rectangular!(model, data, T)

    return model
end

function get_old_voltages(model::JuMP.Model, data::MatpowerData)
    node_vr = Dict()
    node_vi = Dict()
    
    for (i, bus) in data.buses
        node_vr[i] = value.(model[:vr][i,1])
        node_vi[i] = value.(model[:vi][i,1])
    end

    return node_vr, node_vi
end



function _parse_file_data(file_path::String)::MatpowerData
    # 1. Parse the data
    data = PowerModels.parse_file(file_path)
    
    # Standardize data (adds thermal limits if missing, makes costs uniform)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)

    return MatpowerData(data["bus"], data["gen"], data["branch"], data["load"], get(data, "shunt", Dict{String, Any}()))
end


function _add_acuc_var_polar!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64})
    """
    Adds mulit-period ACUC variables
    """
    # multi period opf
    # buses, gens, branches, _ = _unpack_matpowerdata(data)
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

    # 3. Define Variables
    @variable(model, va[keys(buses), T]) # Voltage angle
    @variable(model, buses[i]["vmin"] <= vm[i in keys(buses), T] <= buses[i]["vmax"]) # Voltage magnitude
    
    @variable(model, u[keys(gens), T], Bin) # UNIT COMMITMENT: Binary status
    @variable(model, pg[keys(gens), T])     # Active power generation
    @variable(model, qg[keys(gens), T])     # Reactive power generation

    # Branch flow variables (from and to ends)
    @variable(model, p_fr[keys(branches), T])
    @variable(model, q_fr[keys(branches), T])
    @variable(model, p_to[keys(branches), T])
    @variable(model, q_to[keys(branches), T])
end

function _get_edges(data::MatpowerData)
    # edges = [(string(branch["f_bus"]), string(branch["t_bus"])) for (i, branch) in data.branches]
    edges = [(branch["f_bus"], branch["t_bus"]) for (i, branch) in data.branches]
    return edges
end

function _add_acuc_var_rectangular!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64}; convex::Bool=false, relax_binary::Bool=false, include_uc::Bool=true)
    """
    Rectangular power voltage W-matrix style formulation
    """
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

    # 3. Define Variables
    @variable(model, vr[i in keys(buses), T]) # real voltage
    @variable(model, vi[i in keys(buses), T]) # imaginary voltage
    @variable(model, c_ii[i in keys(buses), T])

    edges = _get_edges(data) # vector of tuples listing the connections
    for (i,bus) in buses
        @constraint(model, [t in T], bus["vmin"]^2 <= model[:c_ii][i,t] <= bus["vmax"]^2) #1i
    end
    # for (i, j) in edges
        # define only for edges
    @variable(model, c_ij[e in edges,T])
    @variable(model, s_ij[e in edges,T])

    if !convex
        for (i, bus) in buses
            @constraint(model, [t in T], model[:c_ii][i,t] == model[:vr][i,t]^2 + model[:vi][i,t]^2) #1l
        end

        for (i,j) in edges
            # define eq. constraints for cij, sij
            e = (i,j)
            i = string(i); j = string(j)
            @constraint(model, [t in T], model[:c_ij][e,t] == model[:vr][i,t]*model[:vr][j,t] + model[:vi][i,t]*model[:vi][j,t]) # 1m
            @constraint(model, [t in T], model[:s_ij][e,t] == model[:vr][i,t]*model[:vi][j,t] - model[:vr][j,t]*model[:vi][i,t]) # 1n
        end
    end

    if include_uc
        if relax_binary
            @variable(model, 0 <= u[keys(gens), T] <= 1) # Relaxed unit commitment for DiffOpt
        else
            @variable(model, u[keys(gens), T], Bin) # UNIT COMMITMENT: Binary status
        end
    end
    @variable(model, pg[keys(gens), T])     # Active power generation
    @variable(model, qg[keys(gens), T])     # Reactive power generation

    # Branch flow variables (from and to ends)
    @variable(model, p_fr[keys(branches), T])
    @variable(model, q_fr[keys(branches), T])
    @variable(model, p_to[keys(branches), T])
    @variable(model, q_to[keys(branches), T])
end



function _add_convex_constraints!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64}, node_vr, node_vi)
    """
    Adding the RHS approximation of the nodal voltages Vr_i and Vi_i
    These are the novelty of the Constante-Flores and Li 2026 work
    Constraints 4b to 4j here

    * node_voltages is a dictionary that contains the mapping of the real and imaginary voltages for each node
    """

    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)
    edges = _get_edges(data) # vector of tuples listing the connections

    @variable(model, xi_c[i in keys(buses),T] >= 0) # slack, 4h
    @variable(model, xij_c[e in edges,T] >= 0) # slack, 4i
    @variable(model, xij_s[e in edges,T] >= 0) # slack, 4j

    for (i, bus) in buses
        @constraint(model, [t in T], model[:c_ii][i, t] >= model[:vr][i,t]^2 + model[:vi][i,t]^2) #4b
        @constraint(model, [t in T], model[:c_ii][i, t] <= 
            2*(node_vr[i] * model[:vr][i,t] + node_vi[i] * model[:vi][i,t]) -
            (node_vr[i]^2 + node_vi[i]^2) + xi_c[i,t]) #4c
    end

    for (i,j) in edges, t in T
        e = (i,j)
        i = string(i); j = string(j)
        vr_i = model[:vr][i,t]; vr_j = model[:vr][j,t]
        vi_i = model[:vi][i,t]; vi_j = model[:vi][j,t]
        xij_c = model[:xij_c][e, t]; xij_s = model[:xij_s][e, t]
        c_ij = model[:c_ij][e,t]; s_ij = model[:s_ij][e,t]

        @constraint(model, (vr_i + vr_j)^2 + (vi_i + vi_j)^2 + 4*c_ij <=
            xij_c + 2*(vr_i - vr_j)*(node_vr[i] - node_vr[j]) + 2*(vi_i - vi_j)*(node_vi[i] - node_vi[j]) - 
            ((node_vr[i] - node_vr[j])^2 + (node_vi[i] - node_vi[j])^2)) # 4d

        @constraint(model, (vr_i - vr_j)^2 + (vi_i - vi_j)^2 + 4*c_ij <=
            xij_c + 2*(vr_i + vr_j)*(node_vr[i] + node_vr[j]) + 2*(vi_i + vi_j)*(node_vi[i] + node_vi[j]) - 
            ((node_vr[i] + node_vr[j])^2 + (node_vi[i] + node_vi[j])^2)) # 4e
             
        @constraint(model, (vr_i - vi_j)^2 + (vr_j + vi_i)^2 + 4*s_ij <=
            xij_s + 2*(vr_i + vi_j)*(node_vr[i] + node_vi[j]) + 2*(vr_j - vi_i)*(node_vr[j] + node_vi[i]) - 
            ((node_vr[i] + node_vi[j])^2 + (node_vr[j] - node_vi[i])^2)) # 4f

        @constraint(model, (vr_i + vi_j)^2 + (vr_j - vi_i)^2 - 4*s_ij <=
            xij_s + 2*(vr_i - vi_j)*(node_vr[i] - node_vi[j]) + 2*(vr_j + vi_i)*(node_vr[j] + node_vi[i]) - 
            ((node_vr[i] - node_vi[j])^2 + (node_vr[j] + node_vi[i])^2)) # 4f
    end

end

function _add_mincost_obj!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64}; convex::Bool=false, no_uc::Bool=false)
    """
    Add min cost objective function for mp-ac-uc-opf.
    Set no_uc=true to exclude the no-load cost term (cost[3] * u) when UC
    variables are not present in the model (e.g. build_diff_opf).
    """
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

    if convex
        edges = _get_edges(data)
        if no_uc
            @objective(model, Min, sum( sum(
                gens[i]["cost"][1] * model[:pg][i, t]^2 +
                gens[i]["cost"][2] * model[:pg][i, t] for i in keys(gens)) + sum(
                model[:xi_c][i,t] for i in keys(buses)) + sum(model[:xij_c][e,t] + model[:xij_s][e,t]
                for e in edges)
                for t in T)
                )
        else
            @objective(model, Min, sum( sum(
                gens[i]["cost"][1] * model[:pg][i, t]^2 +
                gens[i]["cost"][2] * model[:pg][i, t] +
                gens[i]["cost"][3] * model[:u][i, t] for i in keys(gens)) + sum(
                model[:xi_c][i,t] for i in keys(buses)) + sum(model[:xij_c][e,t] + model[:xij_s][e,t]
                for e in edges)
                for t in T)
                )
        end
    else
        @objective(model, Min, sum( sum(
            gens[i]["cost"][1] * model[:pg][i, t]^2 +
            gens[i]["cost"][2] * model[:pg][i, t] +
            gens[i]["cost"][3] * model[:u][i, t] for i in keys(gens)
            ) for t in T))
    end
end


function _add_ref_limits_polar!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64})
    # buses, _, _, _ = _unpack_matpowerdata(data)
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

    # Reference Bus Angle
    ref_buses = [k for (k,v) in buses if v["bus_type"] == 3]
    for t in T, i in ref_buses
        @constraint(model, model[:va][i, t] == 0.0)
    end
end

function _add_ref_limits_rectangular!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64})
    ref_buses = [k for (k,v) in data.buses if v["bus_type"] == 3]
    for t in T, i in ref_buses
        @constraint(model, model[:vi][i, t] == 0.0)
        @constraint(model, model[:vr][i, t] >= 0.0) # Keeps the solution in the positive real half-plane
    end
end

function _add_gen_limits!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64})

    # _, gens, _, _ = _unpack_matpowerdata(data)
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

    # Generator Operational Limits tied to Commitment Status
    for t in T, (i, gen) in gens
        @constraint(model, model[:pg][i,t] >= gen["pmin"] * model[:u][i,t])
        @constraint(model, model[:pg][i,t] <= gen["pmax"] * model[:u][i,t])
        @constraint(model, model[:qg][i,t] >= gen["qmin"] * model[:u][i,t])
        @constraint(model, model[:qg][i,t] <= gen["qmax"] * model[:u][i,t])
    end
end

function _add_continuous_gen_limits!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64})
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

    # Simple generator bounds without UC coupling — used in build_diff_opf
    # where u is excluded so the OPF layer is a pure continuous convex QCQP
    for t in T, (i, gen) in gens
        @constraint(model, model[:pg][i,t] >= gen["pmin"])
        @constraint(model, model[:pg][i,t] <= gen["pmax"])
        @constraint(model, model[:qg][i,t] >= gen["qmin"])
        @constraint(model, model[:qg][i,t] <= gen["qmax"])
    end
end

function _add_polar_branchflow!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64})
    """
    Add branch flows in polar coordinates
    """
    # _, _, branches, _ = _unpack_matpowerdata(data)
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

    # Branch Power Flow Constraints (AC Polar Formulation)
    for t in T, (i, branch) in branches
        f_bus = string(branch["f_bus"])
        t_bus = string(branch["t_bus"])
        
        # Calculate admittance and tap ratios
        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)
        g_fr = branch["g_fr"]; b_fr = branch["b_fr"]
        g_to = branch["g_to"]; b_to = branch["b_to"]
        tm = branch["tap"]
        
        # From-side flows
        @NLconstraint(model, model[:p_fr][i,t] ==  (g + g_fr)/tm^2 * model[:vm][f_bus,t]^2 + 
            (-g*tr + b*ti)/tm^2 * (model[:vm][f_bus,t] * model[:vm][t_bus,t] * cos(model[:va][f_bus,t] - model[:va][t_bus,t])) + 
            (-b*tr - g*ti)/tm^2 * (model[:vm][f_bus,t] * model[:vm][t_bus,t] * sin(model[:va][f_bus,t] - model[:va][t_bus,t])))
            
        @NLconstraint(model, model[:q_fr][i,t] == -(b + b_fr)/tm^2 * model[:vm][f_bus,t]^2 - 
            (-b*tr - g*ti)/tm^2 * (model[:vm][f_bus,t] * model[:vm][t_bus,t] * cos(model[:va][f_bus,t] - model[:va][t_bus,t])) + 
            (-g*tr + b*ti)/tm^2 * (model[:vm][f_bus,t] * model[:vm][t_bus,t] * sin(model[:va][f_bus,t] - model[:va][t_bus,t])))

        # To-side flows
        @NLconstraint(model, model[:p_to][i,t] ==  (g + g_to) * model[:vm][t_bus,t]^2 + 
            (-g*tr - b*ti)/tm^2 * (model[:vm][t_bus,t] * model[:vm][f_bus,t] * cos(model[:va][t_bus,t] - model[:va][f_bus,t])) + 
            (-b*tr + g*ti)/tm^2 * (model[:vm][t_bus,t] * model[:vm][f_bus,t] * sin(model[:va][t_bus,t] - model[:va][f_bus,t])))
            
        @NLconstraint(model, model[:q_to][i,t] == -(b + b_to) * model[:vm][t_bus,t]^2 - 
            (-b*tr + g*ti)/tm^2 * (model[:vm][t_bus,t] * model[:vm][f_bus,t] * cos(model[:va][t_bus,t] - model[:va][f_bus,t])) + 
            (-g*tr - b*ti)/tm^2 * (model[:vm][t_bus,t] * model[:vm][f_bus,t] * sin(model[:va][t_bus,t] - model[:va][f_bus,t])))
            
        # Thermal Limits
        @constraint(model, model[:p_fr][i,t]^2 + model[:q_fr][i,t]^2 <= branch["rate_a"]^2)
        @constraint(model, model[:p_to][i,t]^2 + model[:q_to][i,t]^2 <= branch["rate_a"]^2)
    end
end



function _add_rectangular_branchflow!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64})
    for t in T, (i, branch) in data.branches
        f_bus = string(branch["f_bus"])
        t_bus = string(branch["t_bus"])
        
        # Calculate admittance components
        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch) # Tap ratio and shift
        g_fr, b_fr = branch["g_fr"], branch["b_fr"]
        g_to, b_to = branch["g_to"], branch["b_to"]
        tm2 = branch["tap"]^2
        
        # Access variables for readability
        c_ii = model[:c_ii][f_bus, t]
        c_ij = model[:c_ij][(parse(Int64,f_bus), parse(Int64,t_bus)), t]
        s_ij = model[:s_ij][(parse(Int64,f_bus), parse(Int64,t_bus)), t]
        # p_fr = model[:p_fr][]
        p_fr, q_fr = model[:p_fr][i, t], model[:q_fr][i, t]
        p_to, q_to = model[:p_to][i, t], model[:q_to][i, t]

        # --- From-side power flow (Quadratic) ---
        # Real: P_fr = (g+g_fr)/tm^2 * (vr_f^2 + vi_f^2) + ...
        @constraint(model, p_fr == (g + g_fr)/tm2 * c_ii + 
            (-g*tr + b*ti)/tm2 * c_ij + 
            (-b*tr - g*ti)/tm2 * s_ij)

        # Reactive: Q_fr = -(b+b_fr)/tm^2 * (vr_f^2 + vi_f^2) + ...
        @constraint(model, q_fr == -(b + b_fr)/tm2 * c_ii - 
            (-b*tr - g*ti)/tm2 * c_ij + 
            (-g*tr + b*ti)/tm2 * s_ij)

        # --- To-side power flow (Quadratic) ---
        @constraint(model, p_to == (g + g_to) * c_ii + 
            (-g*tr - b*ti)/tm2 * c_ij + 
            (-b*tr + g*ti)/tm2 * s_ij)

        @constraint(model, q_to == -(b + b_to) * c_ii - 
            (-b*tr + g*ti)/tm2 * c_ij + 
            (-g*tr - b*ti)/tm2 * s_ij)

        # Thermal Limits (Quadratic)
        @constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        @constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
    end
end

function _add_node_bal_polar!(model::JuMP.Model, data::MatpowerData, demand_curve::Vector{Float64})

    # buses, gens, branches, loads, shunts = _unpack_matpowerdata(data)
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

    # Nodal Power Balance (Kirchhoff's Current Law)
    for (t, val) in enumerate(demand_curve), (i, bus) in buses
        # Find all components connected to this bus
        bus_loads = [l for (k,l) in loads if string(l["load_bus"]) == i]
        bus_gens = [k for (k,g) in gens if string(g["gen_bus"]) == i]
        br_fr = [k for (k,b) in branches if string(b["f_bus"]) == i]
        br_to = [k for (k,b) in branches if string(b["t_bus"]) == i]

        pd = sum(l["pd"] for l in bus_loads; init=0.0)*val
        qd = sum(l["qd"] for l in bus_loads; init=0.0)*val
        
        local gs
        local bs
        # println(shunts[i])
        if haskey(shunts, i) # if there exists gs, bs
            println(shunts[i])
            gs = shunts[i]["gs"]; bs = shunts[i]["bs"]
        else
            gs = 0; bs = 0
        end

        @constraint(model, 
            sum(model[:pg][g, t] for g in bus_gens; init=0.0) - pd - gs * model[:vm][i, t]^2 == 
            sum(model[:p_fr][b, t] for b in br_fr; init=0.0) + sum(model[:p_to][b, t] for b in br_to; init=0.0)
        )
        @constraint(model, 
            sum(model[:qg][g, t] for g in bus_gens; init=0.0) - qd + bs * model[:vm][i, t]^2 == 
            sum(model[:q_fr][b, t] for b in br_fr; init=0.0) + sum(model[:q_to][b, t] for b in br_to; init=0.0)
        )
    end
end


function _add_node_bal_rectangular!(model::JuMP.Model, data::MatpowerData, demand_curve::Vector{Int64}; param_loads=nothing)
    # buses, gens, branches, loads, shunts = _unpack_matpowerdata(data)
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

    # Nodal Power Balance (Kirchhoff's Current Law)
    for (t, val) in enumerate(demand_curve), (i, bus) in buses
        # Find all components connected to this bus
        bus_loads = [l for (k,l) in loads if string(l["load_bus"]) == i]
        bus_gens = [k for (k,g) in gens if string(g["gen_bus"]) == i]
        br_fr = [k for (k,b) in branches if string(b["f_bus"]) == i]
        br_to = [k for (k,b) in branches if string(b["t_bus"]) == i]


        if isnothing(param_loads)
            pd = sum(l["pd"] for l in bus_loads; init=0.0)*val
            qd = sum(l["qd"] for l in bus_loads; init=0.0)*val
        else
            pd = param_loads["pd"][i]
            qd = param_loads["qd"][i]
        end
        
        gs = get(bus, "gs", 0.0) + sum(get(s, "gs", 0.0) for (k,s) in shunts if string(s["shunt_bus"]) == i; init=0.0)
        bs = get(bus, "bs", 0.0) + sum(get(s, "bs", 0.0) for (k,s) in shunts if string(s["shunt_bus"]) == i; init=0.0)


        @constraint(model, 
            sum(model[:pg][g, t] for g in bus_gens; init=0.0) - pd - gs * model[:c_ii][i, t] == 
            sum(model[:p_fr][b, t] for b in br_fr; init=0.0) + sum(model[:p_to][b, t] for b in br_to; init=0.0)
        )
        @constraint(model, 
            sum(model[:qg][g, t] for g in bus_gens; init=0.0) - qd + bs * model[:c_ii][i, t] == 
            sum(model[:q_fr][b, t] for b in br_fr; init=0.0) + sum(model[:q_to][b, t] for b in br_to; init=0.0)
        )
    end
end

end
