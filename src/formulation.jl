# Optimization Problem Structure
using JuMP, DiffOpt, Ipopt, Gurobi

using PowerModels

# Surrogate Model Architecture
using Flux

"""
Here is where we will maintain the problem structure and the core functions to running this problem.
"""

struct TrainingData
    u_bar::Vector{Float64}
    x_star::Vector{Float64}
end  

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


function _parse_file_data(file_path::String)::MatpowerData
    # 1. Parse the data
    data = PowerModels.parse_file(file_path)
    
    # Standardize data (adds thermal limits if missing, makes costs uniform)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)

    return MatpowerData(data["bus"], data["gen"], data["branch"], data["load"], get(data, "shunt", Dict{String, Any}()))
end

# function _unpack_matpowerdata(data::MatpowerData)
#     buses = data.buses
#     gens = data.gens
#     branches = data.branches
#     loads = data.loads
#     shunts = data.shunts
#     # return buses, gens, branches, loads
# end

# function _unpack_matpowerdatashunt(data::MatpowerDataShunt)
#     buses = data.buses
#     gens = data.gens
#     branches = data.branches
#     loads = data.loads
#     shunts = data.shunts

#     return buses, gens, branches, loads, shunts
# end



# function _add_acuc_var_polar!(model::JuMP.Model, data::MatpowerData)
#     """
#     Adds single period ACUC variables
#     If mp == true, add multi-period OPF formulation variables
#     """
#     # single period opf
#     # buses, gens, branches, _ = _unpack_matpowerdata(data)
#     buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

#     # 3. Define Variables
#     @variable(model, va[keys(buses)]) # Voltage angle
#     @variable(model, buses[i]["vmin"] <= vm[i in keys(buses)] <= buses[i]["vmax"]) # Voltage magnitude
    
#     @variable(model, u[keys(gens)], Bin) # UNIT COMMITMENT: Binary status
#     @variable(model, pg[keys(gens)])     # Active power generation
#     @variable(model, qg[keys(gens)])     # Reactive power generation

#     # Branch flow variables (from and to ends)
#     @variable(model, p_fr[keys(branches)])
#     @variable(model, q_fr[keys(branches)])
#     @variable(model, p_to[keys(branches)])
#     @variable(model, q_to[keys(branches)])
# end


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

function _add_acuc_var_rectangular!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64})
    """
    Adds multi-period ACUC variables for rectangular coords
    """
    # multi period opf
    # buses, gens, branches, _ = _unpack_matpowerdata(data)
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

    # 3. Define Variables
    @variable(model, vr[i in keys(data.buses), T]) # real voltage
    @variable(model, vi[i in keys(data.buses), T]) # imaginary voltage

    for (i, bus) in data.buses
        @constraint(model, [t in T], model[:vr][i,t]^2 + model[:vi][i,t]^2 <= bus["vmax"]^2)
        @constraint(model, [t in T], model[:vr][i,t]^2 + model[:vi][i,t]^2 >= bus["vmin"]^2)
    end
    
    @variable(model, u[keys(gens), T], Bin) # UNIT COMMITMENT: Binary status
    @variable(model, pg[keys(gens), T])     # Active power generation
    @variable(model, qg[keys(gens), T])     # Reactive power generation

    # Branch flow variables (from and to ends)
    @variable(model, p_fr[keys(branches), T])
    @variable(model, q_fr[keys(branches), T])
    @variable(model, p_to[keys(branches), T])
    @variable(model, q_to[keys(branches), T])
end


# function _add_mincost_obj!(model::JuMP.Model, data::MatpowerData)
#     """
#     Add min cost objective function
#     """

#     # _, gens, _, _ = _unpack_matpowerdata(data)
#     buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

#     @objective(model, Min, sum(
#         gens[i]["cost"][1] * model[:pg][i]^2 + 
#         gens[i]["cost"][2] * model[:pg][i] + 
#         gens[i]["cost"][3] * model[:u][i] for i in keys(gens)
#     ))
# end


function _add_mincost_obj!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64})
    """
    Add min cost objective function for mp-ac-uc-opf
    """


    # _, gens, _, _ = _unpack_matpowerdata(data)
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)


    @objective(model, Min, sum( sum(
        gens[i]["cost"][1] * model[:pg][i, t]^2 + 
        gens[i]["cost"][2] * model[:pg][i, t] + 
        gens[i]["cost"][3] * model[:u][i, t] for i in keys(gens)
        ) for t in T))
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
        vr_f, vi_f = model[:vr][f_bus, t], model[:vi][f_bus, t]
        vr_t, vi_t = model[:vr][t_bus, t], model[:vi][t_bus, t]
        p_fr, q_fr = model[:p_fr][i, t], model[:q_fr][i, t]
        p_to, q_to = model[:p_to][i, t], model[:q_to][i, t]

        # --- From-side power flow (Quadratic) ---
        # Real: P_fr = (g+g_fr)/tm^2 * (vr_f^2 + vi_f^2) + ...
        @constraint(model, p_fr == (g + g_fr)/tm2 * (vr_f^2 + vi_f^2) + 
            (-g*tr + b*ti)/tm2 * (vr_f*vr_t + vi_f*vi_t) + 
            (-b*tr - g*ti)/tm2 * (vi_f*vr_t - vr_f*vi_t))

        # Reactive: Q_fr = -(b+b_fr)/tm^2 * (vr_f^2 + vi_f^2) + ...
        @constraint(model, q_fr == -(b + b_fr)/tm2 * (vr_f^2 + vi_f^2) - 
            (-b*tr - g*ti)/tm2 * (vr_f*vr_t + vi_f*vi_t) + 
            (-g*tr + b*ti)/tm2 * (vi_f*vr_t - vr_f*vi_t))

        # --- To-side power flow (Quadratic) ---
        @constraint(model, p_to == (g + g_to) * (vr_t^2 + vi_t^2) + 
            (-g*tr - b*ti)/tm2 * (vr_t*vr_f + vi_t*vi_f) + 
            (-b*tr + g*ti)/tm2 * (vi_t*vr_f - vr_t*vi_f))

        @constraint(model, q_to == -(b + b_to) * (vr_t^2 + vi_t^2) - 
            (-b*tr + g*ti)/tm2 * (vr_t*vr_f + vi_t*vi_f) + 
            (-g*tr - b*ti)/tm2 * (vi_t*vr_f - vr_t*vi_f))

        # Thermal Limits (Quadratic)
        @constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        @constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
    end
end


function _add_node_bal_polar!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64})

    # buses, gens, branches, loads, shunts = _unpack_matpowerdata(data)
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

    # Nodal Power Balance (Kirchhoff's Current Law)
    for t in T, (i, bus) in buses
        # Find all components connected to this bus
        bus_loads = [l for (k,l) in loads if string(l["load_bus"]) == i]
        bus_gens = [k for (k,g) in gens if string(g["gen_bus"]) == i]
        br_fr = [k for (k,b) in branches if string(b["f_bus"]) == i]
        br_to = [k for (k,b) in branches if string(b["t_bus"]) == i]

        pd = sum(l["pd"] for l in bus_loads; init=0.0)
        qd = sum(l["qd"] for l in bus_loads; init=0.0)
        
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


function _add_node_bal_rectangular!(model::JuMP.Model, data::MatpowerData, T::Vector{Int64})

    # buses, gens, branches, loads, shunts = _unpack_matpowerdata(data)
    buses, gens, branches, loads, shunts = (data.buses, data.gens, data.branches, data.loads, data.shunts)

    # Nodal Power Balance (Kirchhoff's Current Law)
    for t in T, (i, bus) in buses
        # Find all components connected to this bus
        bus_loads = [l for (k,l) in loads if string(l["load_bus"]) == i]
        bus_gens = [k for (k,g) in gens if string(g["gen_bus"]) == i]
        br_fr = [k for (k,b) in branches if string(b["f_bus"]) == i]
        br_to = [k for (k,b) in branches if string(b["t_bus"]) == i]

        pd = sum(l["pd"] for l in bus_loads; init=0.0)
        qd = sum(l["qd"] for l in bus_loads; init=0.0)
        
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
            sum(model[:pg][g, t] for g in bus_gens; init=0.0) - pd - gs * (model[:vr][i, t]^2 + model[:vi][i, t]^2) == 
            sum(model[:p_fr][b, t] for b in br_fr; init=0.0) + sum(model[:p_to][b, t] for b in br_to; init=0.0)
        )
        @constraint(model, 
            sum(model[:qg][g, t] for g in bus_gens; init=0.0) - qd + bs * (model[:vr][i, t]^2 + model[:vi][i, t]^2) == 
            sum(model[:q_fr][b, t] for b in br_fr; init=0.0) + sum(model[:q_to][b, t] for b in br_to; init=0.0)
        )
    end
end


function build_single_period_ac_uc_polar(file_path::String)
    
    data = _parse_file_data(file_path)
    T = [1]
    # data_s = _parse_file_data_shunt(file_path)

    # Create dictionaries for easy iteration
    # buses = data.buses
    # gens = data.gens
    # branches = data.branches
    # loads = data.loads

    # Initialize the JuMP Model
    model = Model(Gurobi.Optimizer)

    # Add variables 
    _add_acuc_var_polar!(model, data, T)

    # Add objective
    _add_mincost_obj!(model, data, T)

    # Add constraints
    _add_ref_limits_polar!(model, data, T)

    _add_gen_limits!(model, data, T)

    _add_polar_branchflow!(model, data, [1])

    _add_node_bal_polar!(model, data, T)

    return model
end


function build_single_period_ac_uc_rectangular(file_path::String)
    
    data = _parse_file_data(file_path)
    T = [1]
    # data_s = _parse_file_data_shunt(file_path)

    # Create dictionaries for easy iteration
    # buses = data.buses
    # gens = data.gens
    # branches = data.branches
    # loads = data.loads

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


# function build_multi_period_ac_uc(file_path::String, demand_curve::Vector{Real})
#     """
#     * file_path::String : path to the .m file containing the OPF data
#     * demand_curve::Vector{Int64} : vector to define the scaled baseline demand to generate a demand curve
#     Akin to ExaModelsPower.jl
#     """
    
    
#     data = PowerModels.parse_file(file_path)
#     PowerModels.standardize_cost_terms!(data, order=2)
#     PowerModels.calc_thermal_limits!(data)

#     # Injecting dummy UC parameters since standard .m files lack them
#     for (i, gen) in data["gen"]
#         gen["ramp_up"] = gen["pmax"] * 0.3    # Can ramp 30% of max capacity per hour
#         gen["ramp_down"] = gen["pmax"] * 0.3
#         gen["startup_cost"] = 500.0
#     end

#     T = length(demand_curve)

#     model = Model()
#     buses = data["bus"]; gens = data["gen"]; branches = data["branch"]; loads = data["load"]

#     # --- Time-Indexed Variables ---
#     @variable(model, va[keys(buses), 1:T]) 
#     @variable(model, buses[i]["vmin"] <= vm[i in keys(buses), t in 1:T] <= buses[i]["vmax"])
    
#     @variable(model, u[keys(gens), 1:T], Bin) # On/Off status
#     @variable(model, v[keys(gens), 1:T], Bin) # Startup status
#     @variable(model, w[keys(gens), 1:T], Bin) # Shutdown status
    
#     @variable(model, pg[keys(gens), 1:T])
#     @variable(model, qg[keys(gens), 1:T])

#     @variable(model, p_fr[keys(branches), 1:T])
#     @variable(model, q_fr[keys(branches), 1:T])
#     @variable(model, p_to[keys(branches), 1:T])
#     @variable(model, q_to[keys(branches), 1:T])

#     # --- Objective (Includes startup costs) ---
#     @objective(model, Min, sum(
#         sum(gens[i]["cost"][1] * pg[i, t]^2 + 
#             gens[i]["cost"][2] * pg[i, t] + 
#             gens[i]["cost"][3] * u[i, t] + 
#             gens[i]["startup_cost"] * v[i, t] for i in keys(gens)) 
#         for t in 1:T
#     ))

#     # --- Constraints ---
#     for t in 1:T
#         # 1. Network Physics (AC flows & Nodal Balance) for each time period
#         # (This uses the exact same logic as Part 1, just adding the `t` index)
#         # For brevity in this block, assume the same KCL and AC flow equations as above, 
#         # but applied to va[i, t], vm[i, t], pg[i, t], etc.
        
#         for (i, gen) in gens
#             # 2. Generator Limits
#             @constraint(model, pg[i, t] >= gen["pmin"] * u[i, t])
#             @constraint(model, pg[i, t] <= gen["pmax"] * u[i, t])

#             # 3. Inter-temporal UC Logic (Ramping & State tracking)
#             if t > 1
#                 # Logical relationship between state, startup, and shutdown
#                 @constraint(model, u[i, t] - u[i, t-1] == v[i, t] - w[i, t])
                
#                 # Ramping Limits
#                 @constraint(model, pg[i, t] - pg[i, t-1] <= gen["ramp_up"] * u[i, t-1] + gen["pmin"] * v[i, t])
#                 @constraint(model, pg[i, t-1] - pg[i, t] <= gen["ramp_down"] * u[i, t] + gen["pmin"] * w[i, t])
#             else
#                 # Initial condition (Assuming all generators start OFF at t=0 for simplicity)
#                 @constraint(model, u[i, t] == v[i, t])
#             end
#         end
#     end

#     return model, data
# end



function solve_monolithic(model::JuMP.Model)
    """
    Function to gather training data for the convex MIQCQP.
        model - JuMP model containing the monolithic MIQCP
    Returns:
        training_data - mapping of the fixed parameters \bar{u} --> x*
    """
    optimize!(model)
    
    u = value.(model[:u])
    x = value.(model[:x])
    
    return TrainingData(u, x)
end