module surrogates

using DiffOpt
using Flux
using JuMP
using JLD2
using PowerModels
using ProgressMeter
using Random

include(joinpath(@__DIR__, "formulation.jl"))
using .formulation

"""
Stores one solved instance of the convex AC OPF+UC problem.

Fields:
- `node_vr`: real voltage parameters used as the linearization point (input ξ)
- `node_vi`: imaginary voltage parameters used as the linearization point (input ξ)
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


"""
    sample_training_data(file_name, n_samples; save_path) -> Vector{TrainingDataPoint}

Generate `n_samples` training points by Latin Hypercube Sampling over per-bus
voltage bounds and solving the convex AC OPF+UC for each sample.

Bounds (derived from Matpower bus data):
- `node_vr[i]` ∈ `[vmin[i], vmax[i]]`
- `node_vi[i]` ∈ `[-vmax[i], vmax[i]]`

If `save_path` is provided the resulting repertoire is saved (appended) to that
JLD2 file via `save_repertoire`.

Failed solves are retained in the output with an empty `x_opt` so the caller
can inspect or filter them.
"""
function sample_training_data(
    file_name::String,
    n_samples::Int;
    save_path::Union{String, Nothing} = nothing,
)::Vector{TrainingDataPoint}

    # Parse bus data to extract per-bus voltage bounds
    raw = PowerModels.parse_file(file_name)
    PowerModels.standardize_cost_terms!(raw, order=2)
    PowerModels.calc_thermal_limits!(raw)
    buses = raw["bus"]

    bus_ids = sort(collect(keys(buses)), by = k -> parse(Int, k))
    n_buses = length(bus_ids)
    n_dims  = 2 * n_buses  # vr dims first, then vi dims

    # [vr_lb..., vi_lb...]  and  [vr_ub..., vi_ub...]
    lb = Float64[
        [buses[i]["vmin"] for i in bus_ids]...,   # vr lower
        [-buses[i]["vmax"] for i in bus_ids]...,  # vi lower
    ]
    ub = Float64[
        [buses[i]["vmax"] for i in bus_ids]...,   # vr upper
        [buses[i]["vmax"] for i in bus_ids]...,   # vi upper
    ]

    # Latin Hypercube Sampling — each dimension stratified independently
    lhs = Matrix{Float64}(undef, n_samples, n_dims)
    for d in 1:n_dims
        perm = randperm(n_samples)
        for s in 1:n_samples
            ξ_s = (perm[s] - 1 + rand()) / n_samples   # uniform sample within stratum
            lhs[s, d] = lb[d] + (ub[d] - lb[d]) * ξ_s
        end
    end

    # Solve one instance per sample row
    points = TrainingDataPoint[]
    sizehint!(points, n_samples)

    @showprogress "Sampling training data: " for s in 1:n_samples
        node_vr = Dict(bus_ids[k] => lhs[s, k]           for k in 1:n_buses)
        node_vi = Dict(bus_ids[k] => lhs[s, n_buses + k] for k in 1:n_buses)
        push!(points, gather_training_data(file_name, node_vr, node_vi))
    end

    isnothing(save_path) || save_repertoire(points, save_path)

    return points
end


# ---------------------------------------------------------------------------
# Repertoire: saving and loading collections of TrainingDataPoints
# ---------------------------------------------------------------------------

"""
    save_repertoire(points, path)

Append `points` (a `Vector{TrainingDataPoint}`) to a JLD2 repertoire file at `path`.
If the file already exists, the new points are merged with the existing ones.
"""
function save_repertoire(points::Vector{TrainingDataPoint}, path::String)
    existing = isfile(path) ? load_repertoire(path) : TrainingDataPoint[]
    merged = vcat(existing, points)
    jldsave(path; repertoire=merged)
    println("Saved $(length(merged)) total points to $path ($(length(points)) new).")
end

"""
    load_repertoire(path)

Load a `Vector{TrainingDataPoint}` from a JLD2 repertoire file at `path`.
"""
function load_repertoire(path::String)::Vector{TrainingDataPoint}
    return load(path, "repertoire")
end


# ---------------------------------------------------------------------------
# Surrogate neural network: encoding and architecture
# ---------------------------------------------------------------------------

"""
    encode_ξ(point) -> Vector{Float32}

Flatten `(node_vr, node_vi)` from a `TrainingDataPoint` into the 1-D surrogate
input vector ξ. Buses are ordered by numeric ID.

Layout: `[vr["1"], …, vr["n"], vi["1"], …, vi["n"]]`
"""
function encode_ξ(point::TrainingDataPoint)::Vector{Float32}
    bus_ids = sort(collect(keys(point.node_vr)), by=k -> parse(Int, k))
    return Float32[
        [point.node_vr[i] for i in bus_ids]...,
        [point.node_vi[i] for i in bus_ids]...,
    ]
end

"""
    encode_x(point) -> Vector{Float32}

Flatten `x_opt` from a `TrainingDataPoint` into a 1-D target vector.
Keys within each variable group are sorted for a consistent layout across all points.

Layout (groups in order):
`vr, vi, c_ii` (per bus) | `c_ij, s_ij` (per edge) |
`u, pg, qg` (per gen) | `p_fr, q_fr, p_to, q_to` (per branch) |
`xi_c` (per bus) | `xij_c, xij_s` (per edge)
"""
function encode_x(point::TrainingDataPoint)::Vector{Float32}
    x        = point.x_opt
    bus_ids  = sort(collect(keys(x["vr"])),    by=k -> parse(Int, k))
    gen_ids  = sort(collect(keys(x["pg"])),    by=k -> parse(Int, k))
    br_ids   = sort(collect(keys(x["p_fr"])), by=k -> parse(Int, k))
    edge_ids = sort(collect(keys(x["c_ij"])))  # (Int,Int) tuples, lexicographic

    return Float32[
        [x["vr"][i]    for i in bus_ids]...,
        [x["vi"][i]    for i in bus_ids]...,
        [x["c_ii"][i]  for i in bus_ids]...,
        [x["c_ij"][e]  for e in edge_ids]...,
        [x["s_ij"][e]  for e in edge_ids]...,
        [x["u"][g]     for g in gen_ids]...,
        [x["pg"][g]    for g in gen_ids]...,
        [x["qg"][g]    for g in gen_ids]...,
        [x["p_fr"][b]  for b in br_ids]...,
        [x["q_fr"][b]  for b in br_ids]...,
        [x["p_to"][b]  for b in br_ids]...,
        [x["q_to"][b]  for b in br_ids]...,
        [x["xi_c"][i]  for i in bus_ids]...,
        [x["xij_c"][e] for e in edge_ids]...,
        [x["xij_s"][e] for e in edge_ids]...,
    ]
end


"""
    build_ffnn(input_dim, output_dim, hidden_dims; activation) -> Chain

Build a feedforward neural network ξ → θ̂.

The FFNN maps the surrogate input ξ = (node_vr, node_vi) to a predicted
linearization point θ̂, which defines the linear cuts A(ξ, θ̂)x ≤ B(ξ, θ̂)
inside the convex OPF layer.

# Arguments
- `input_dim::Int`: length of ξ (output of `encode_ξ`), equal to 2 * n_buses
- `output_dim::Int`: 2 * n_buses  (predicted node_vr and node_vi for each bus)
- `hidden_dims::Vector{Int}`: width of each hidden layer, e.g. `[64, 64]`
- `activation`: activation applied to every hidden layer (default: `relu`)

The output layer is linear (no activation) so the network is unconstrained in range.

# Example
```julia
nn = build_ffnn(28, 28, [64, 64])   # case14: 28 inputs → 28 outputs (node_vr, node_vi)
θ̂  = nn(encode_ξ(point))            # predicted linearization point
```
"""
function build_ffnn(
    input_dim::Int,
    output_dim::Int,
    hidden_dims::Vector{Int};
    activation = relu,
)::Chain
    dims = [input_dim; hidden_dims; output_dim]
    layers = []
    for i in 1:(length(dims) - 2)
        push!(layers, Dense(dims[i], dims[i+1], activation))
    end
    push!(layers, Dense(dims[end-1], dims[end]))  # linear output
    return Chain(layers...)
end


# ---------------------------------------------------------------------------
# (3) Differentiable OPF layer + end-to-end surrogate
# ---------------------------------------------------------------------------
#
# ARCHITECTURE
# ============
#
#   ξ (surrogate input: node voltages, encoded by encode_ξ)
#     │
#     ▼
#   FFNN  (learned weights W)
#     │
#     ▼  θ̂ = (node_vr_pred, node_vi_pred)   shape: 2 × n_buses
#     │
#     │  These define the LINEAR CUTS inside the convex QCQP:
#     │
#     │    4c:  c_ii[i] ≤ 2(θ̂_vr[i]·vr[i] + θ̂_vi[i]·vi[i])
#     │                   - (θ̂_vr[i]² + θ̂_vi[i]²) + ξ_c[i]
#     │    4d–4f: similar bilinear cuts for c_ij, s_ij
#     │
#     │  i.e.  A(ξ, θ̂) x ≤ B(ξ, θ̂)
#     │
#     ▼
#   OPF LAYER  build_diff_opf(file, θ̂_vr, θ̂_vi)   [DiffOpt + ChainRules]
#     │
#     ▼  x* = (vr*, vi*, pg*, u_uc*, ...)   optimal solution under learned cuts
#     │        note: u_uc is the unit commitment decision, distinct from ξ
#     ▼
#   LOSS  L(x*)   e.g. generation cost, or MSE vs a held-out solution
#     │
#     ▼
#   BACKPROP through OPF layer via rrule → dL/dθ̂ → dL/dW_ffnn


# ---------------------------------------------------------------------------
# PSEUDOCODE: differentiable OPF layer
# ---------------------------------------------------------------------------
#
# The OPF layer wraps build_convex_ac_uc so that DiffOpt can differentiate
# through it. The key change vs the current formulation is that node_vr and
# node_vi are declared as Parameter variables (not plain numbers), so DiffOpt
# tracks their sensitivity.
#
# STEP 1 — build a differentiable model
#
#   function build_diff_opf(file_name, θ̂_vr, θ̂_vi)
#       model = DiffOpt.diff_model(Gurobi.Optimizer)
#
#       # Declare linearization point as DiffOpt Parameters
#       @variable(model, node_vr[bus_ids] in Parameter.(θ̂_vr))
#       @variable(model, node_vi[bus_ids] in Parameter.(θ̂_vi))
#
#       # Build the rest of the convex QCQP using node_vr / node_vi
#       # as the cut coefficients  (A(u, θ) x ≤ B(u, θ))
#       _add_acuc_var_rectangular!(model, data, T, convex=true)
#       _add_convex_constraints!(model, data, T, node_vr, node_vi)
#       _add_mincost_obj!(...)
#       ...
#       return model
#   end
#
#
# STEP 2 — define the solution map + rrule
#
#   function opf_layer(file_name, θ̂_vr, θ̂_vi)
#       model = build_diff_opf(file_name, θ̂_vr, θ̂_vi)
#       optimize!(model)
#       return value.(model[:pg])   # or whichever outputs feed the loss
#   end
#
#   function ChainRulesCore.rrule(::typeof(opf_layer), file_name, θ̂_vr, θ̂_vi)
#       model = build_diff_opf(file_name, θ̂_vr, θ̂_vi)
#       x_star = opf_layer(file_name, θ̂_vr, θ̂_vi; model=model)
#
#       function pullback_opf(dL_dx)
#           # seed dL/dx* into DiffOpt
#           DiffOpt.set_reverse_variable.(model, model[:pg], dL_dx)
#           DiffOpt.reverse_differentiate!(model)
#
#           # retrieve dL/dθ̂  (flows back into FFNN)
#           dθ̂_vr = DiffOpt.get_reverse_parameter.(model, model[:node_vr])
#           dθ̂_vi = DiffOpt.get_reverse_parameter.(model, model[:node_vi])
#           return (NoTangent(), NoTangent(), dθ̂_vr, dθ̂_vi)
#       end
#       return x_star, pullback_opf
#   end
#
#
# STEP 3 — end-to-end surrogate model
#
#   function surrogate_forward(ξ_vec, ffnn, file_name, bus_ids)
#       # ξ_vec = encode_ξ(point)  →  shape: 2 * n_buses
#       θ̂ = ffnn(ξ_vec)                        # FFNN predicts linearization point
#       n = length(bus_ids)
#       θ̂_vr = θ̂[1:n];  θ̂_vi = θ̂[n+1:end]   # split into vr / vi
#       x_star = opf_layer(file_name, θ̂_vr, θ̂_vi)
#       return x_star
#   end
#
#
# STEP 4 — training loop
#
#   loss(ξ_vec, x_true) = sum((surrogate_forward(ξ_vec, ffnn, ...) .- x_true).^2)
#
#   opt = Flux.setup(Adam(), ffnn)
#   for (ξ_vec, x_true) in training_data
#       grads = Flux.gradient(ffnn) do nn
#           loss(ξ_vec, x_true)    # gradient flows through OPF layer via rrule
#       end
#       Flux.update!(opt, ffnn, grads)
#   end

end
