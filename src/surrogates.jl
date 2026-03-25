module surrogates

using DiffOpt
using Flux
using JuMP

include("formulation.jl")
using .formulation

"""
Stores one solved instance of the convex AC OPF+UC problem.

Fields:
- `node_vr`: real voltage parameters used as the linearization point (input u)
- `node_vi`: imaginary voltage parameters used as the linearization point (input u)
- `x_opt`: Dict of optimal variable values keyed by variable name
- `status`: solver termination status string
"""
struct TrainingDataPoint
    node_vr::Dict
    node_vi::Dict
    x_opt::Dict
    status::String
end


"""
    gather_training_data(file_name, node_vr, node_vi)

Solve one instance of the convex AC OPF+UC problem and return a `TrainingDataPoint`.

# Arguments
- `file_name::String`: path to the Matpower `.m` case file
- `node_vr::Dict`: real voltage linearization point for each bus (e.g. `Dict("1" => 1.0, ...)`)
- `node_vi::Dict`: imaginary voltage linearization point for each bus (e.g. `Dict("1" => 0.0, ...)`)

# Returns
A `TrainingDataPoint` with the optimal solution, or an empty `x_opt` if the solve fails.
"""
function gather_training_data(file_name::String, node_vr::Dict, node_vi::Dict)
    model = build_convex_ac_uc(file_name, node_vr, node_vi)
    optimize!(model)

    status = string(termination_status(model))

    if !(status in ("OPTIMAL", "LOCALLY_SOLVED"))
        return TrainingDataPoint(node_vr, node_vi, Dict(), status)
    end

    t = 1  # build_convex_ac_uc uses a single time period

    x_opt = Dict(
        "vr"    => Dict(i => value(model[:vr][i, t])    for i in axes(model[:vr], 1)),
        "vi"    => Dict(i => value(model[:vi][i, t])    for i in axes(model[:vi], 1)),
        "c_ii"  => Dict(i => value(model[:c_ii][i, t])  for i in axes(model[:c_ii], 1)),
        "c_ij"  => Dict(e => value(model[:c_ij][e, t])  for e in axes(model[:c_ij], 1)),
        "s_ij"  => Dict(e => value(model[:s_ij][e, t])  for e in axes(model[:s_ij], 1)),
        "u"     => Dict(g => value(model[:u][g, t])     for g in axes(model[:u], 1)),
        "pg"    => Dict(g => value(model[:pg][g, t])    for g in axes(model[:pg], 1)),
        "qg"    => Dict(g => value(model[:qg][g, t])    for g in axes(model[:qg], 1)),
        "p_fr"  => Dict(b => value(model[:p_fr][b, t])  for b in axes(model[:p_fr], 1)),
        "q_fr"  => Dict(b => value(model[:q_fr][b, t])  for b in axes(model[:q_fr], 1)),
        "p_to"  => Dict(b => value(model[:p_to][b, t])  for b in axes(model[:p_to], 1)),
        "q_to"  => Dict(b => value(model[:q_to][b, t])  for b in axes(model[:q_to], 1)),
        "xi_c"  => Dict(i => value(model[:xi_c][i, t])  for i in axes(model[:xi_c], 1)),
        "xij_c" => Dict(e => value(model[:xij_c][e, t]) for e in axes(model[:xij_c], 1)),
        "xij_s" => Dict(e => value(model[:xij_s][e, t]) for e in axes(model[:xij_s], 1)),
    )

    return TrainingDataPoint(node_vr, node_vi, x_opt, status)
end


# (2) Embedding the convex QCQP into the NN
# (3) Gradient of the loss function for decision-focused training

end
