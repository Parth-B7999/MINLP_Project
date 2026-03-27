module surrogates

using DiffOpt
using Flux
using JuMP
using JLD2
using PowerModels
using ProgressMeter
using Random
import ChainRulesCore
import MathOptInterface as MOI

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
# ALTERNATIVE: Polynomial parameterization of θ̂ (Dixit 2025 style)
# ---------------------------------------------------------------------------
#
# Instead of an FFNN, θ̂ = (node_vr, node_vi) can be parameterized as a
# low-order polynomial in ξ, following Dixit et al. (2025):
#
#   θ̂_vr[i] = a_vr[i] + b_vr[i]·ξ_vr[i] + c_vr[i]·ξ_vr[i]²
#   θ̂_vi[i] = a_vi[i] + b_vi[i]·ξ_vi[i] + c_vi[i]·ξ_vi[i]²
#
# Parameters: 3 × n_buses × 2 scalars (84 for case14).
# Warm start: a = mean(ξ), b = 1, c = 0  (identity-like initialization).
#
# ADVANTAGES over FFNN:
#   - Far fewer parameters, easier to train for POC
#   - No Flux needed for the predictor itself
#   - Directly comparable to Dixit for validation
#   - Good for testing that the DiffOpt pipeline (rrule, reverse_differentiate!,
#     training loop) is mechanically correct before committing to the FFNN
#
# WHY WE USE THE FFNN INSTEAD:
#   - The MIQCQP solution x_true is DISCONTINUOUS in ξ — the unit commitment
#     pattern (which generators are on/off) changes discretely as ξ varies.
#     A smooth elementwise polynomial cannot represent these regime changes.
#   - The polynomial maps bus i's input to bus i's θ̂ independently. It has
#     no cross-bus interactions, so it cannot learn that bus i's voltage
#     affects the optimal commitment at a distant generator.
#   - The core claim of the surrogate — that a learned convex QCQP can
#     replace the MIQCQP — requires capturing the combinatorial structure of
#     unit commitment. That needs the expressive power of an FFNN.
#
# REVISIT as a debugging baseline if FFNN training has convergence issues.

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
# STEP 1 — build_diff_opf  (implemented in formulation.jl)
# ---------------------------------------------------------------------------
# build_diff_opf(file_path, θ̂_vr, θ̂_vi) wraps the convex QCQP with
# DiffOpt.diff_model and declares node_vr / node_vi as Parameter variables.
# UC decisions are excluded; generator limits are simple continuous bounds.


# ---------------------------------------------------------------------------
# STEP 2 — opf_layer + rrule
# ---------------------------------------------------------------------------

"""
    opf_layer(file_name, θ̂_vr, θ̂_vi; model) -> Vector{Float64}

Solve the differentiable convex OPF given predicted linearization point θ̂.
Returns optimal active power generation pg* as a Vector sorted by generator ID.

Pass a pre-built `model` to avoid rebuilding — used inside the rrule to
share the solved model with the pullback closure.
"""
function opf_layer(
    file_name::String,
    θ̂_vr::Vector{Float64},
    θ̂_vi::Vector{Float64};
    model::Union{JuMP.Model, Nothing} = nothing,
)::Vector{Float64}
    if isnothing(model)
        model = build_diff_opf(file_name, θ̂_vr, θ̂_vi)
    end
    set_silent(model)
    optimize!(model)

    stat = termination_status(model)
    if !(stat in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED))
        error("opf_layer: solver returned $stat")
    end

    t = 1
    gen_ids = sort(collect(axes(model[:pg], 1)), by = k -> parse(Int, k))
    return Float64[value(model[:pg][g, t]) for g in gen_ids]
end


"""
ChainRulesCore.rrule for opf_layer.

Forward: build model, solve, return pg*.

Pullback: receives dL/dpg* from the downstream loss, seeds it into DiffOpt,
runs reverse_differentiate! to solve the KKT sensitivity system, then returns
dL/dθ̂_vr and dL/dθ̂_vi — the gradients that flow back into the FFNN.
"""
function ChainRulesCore.rrule(
    ::typeof(opf_layer),
    file_name::String,
    θ̂_vr::Vector{Float64},
    θ̂_vi::Vector{Float64},
)
    model   = build_diff_opf(file_name, θ̂_vr, θ̂_vi)
    pg_star = opf_layer(file_name, θ̂_vr, θ̂_vi; model = model)

    function pullback_opf(dL_dpg)
        t       = 1
        gen_ids = sort(collect(axes(model[:pg],     1)), by = k -> parse(Int, k))
        bus_ids = sort(collect(axes(model[:node_vr], 1)), by = k -> parse(Int, k))

        # Clear any previously set sensitivities
        DiffOpt.empty_input_sensitivities!(model)

        # Seed dL/dpg* — one scalar per generator, ordered by gen ID
        for (k, g) in enumerate(gen_ids)
            DiffOpt.set_reverse_variable(model, model[:pg][g, t], dL_dpg[k])
        end

        # Solve the KKT sensitivity system through the OPF
        DiffOpt.reverse_differentiate!(model)

        # Retrieve dL/dθ̂ — gradients w.r.t. the linearization point Parameters
        dθ̂_vr = [DiffOpt.get_reverse_parameter(model, model[:node_vr][i]) for i in bus_ids]
        dθ̂_vi = [DiffOpt.get_reverse_parameter(model, model[:node_vi][i]) for i in bus_ids]

        return (
            ChainRulesCore.NoTangent(),  # ∂/∂(typeof(opf_layer))
            ChainRulesCore.NoTangent(),  # ∂/∂file_name
            dθ̂_vr,
            dθ̂_vi,
        )
    end

    return pg_star, pullback_opf
end


# ---------------------------------------------------------------------------
# STEP 3 — end-to-end surrogate forward pass
# ---------------------------------------------------------------------------

"""
    surrogate_forward(ξ_vec, ffnn, file_name, n_buses) -> Vector{Float64}

Chain FFNN → OPF layer for a single input ξ.

# Arguments
- `ξ_vec::Vector{Float32}`: encoded input `[vr_1,...,vr_n, vi_1,...,vi_n]`
- `ffnn::Chain`: Flux network mapping ξ → θ̂  (built with `build_ffnn`)
- `file_name::String`: path to the Matpower `.m` case file
- `n_buses::Int`: number of buses (used to split θ̂ into vr / vi)

# Returns
Optimal active power generation `pg*` as a `Vector{Float64}`, sorted by generator ID.
This is differentiable end-to-end: Zygote traces through the FFNN, and the
`rrule` for `opf_layer` handles the OPF sensitivity via DiffOpt.
"""
function surrogate_forward(
    ξ_vec::Vector{Float32},
    ffnn::Chain,
    file_name::String,
    n_buses::Int,
)::Vector{Float64}
    θ̂     = ffnn(ξ_vec)                     # FFNN: ξ → θ̂  (Float32)
    θ̂_vr  = Float64.(θ̂[1:n_buses])          # split and promote for OPF layer
    θ̂_vi  = Float64.(θ̂[n_buses+1:end])
    return opf_layer(file_name, θ̂_vr, θ̂_vi) # → pg* via DiffOpt rrule
end


# ---------------------------------------------------------------------------
# STEP 4 — training loop
# ---------------------------------------------------------------------------

"""
    train_surrogate!(ffnn, training_data, file_name, bus_ids, gen_ids;
                     n_epochs, lr) -> Vector{Float64}

Train `ffnn` end-to-end using MSE on active power generation as the task loss.

Loss per sample:  L = ∑_g (pg*_g − pg_true_g)²

Gradients flow:
  dL/dpg* → [rrule] → dL/dθ̂ → [Zygote] → dL/dW_ffnn

Failed solves in `training_data` (empty `x_opt`) are skipped.

# Arguments
- `ffnn::Chain`: surrogate network built with `build_ffnn`; mutated in-place
- `training_data::Vector{TrainingDataPoint}`: output of `sample_training_data`
- `file_name::String`: path to Matpower `.m` case file (passed to `opf_layer`)
- `bus_ids::Vector{String}`: sorted bus IDs (determines ξ layout and θ̂ split)
- `gen_ids::Vector{String}`: sorted generator IDs (determines pg* / pg_true layout)
- `n_epochs::Int`: number of full passes over `training_data` (default: 10)
- `lr::Float64`: Adam learning rate (default: 1e-3)

# Returns
`Vector{Float64}` of mean epoch losses (length `n_epochs`).
"""
function train_surrogate!(
    ffnn::Chain,
    training_data::Vector{TrainingDataPoint},
    file_name::String,
    bus_ids::Vector{String},
    gen_ids::Vector{String};
    n_epochs::Int   = 10,
    lr::Float64     = 1e-3,
)::Vector{Float64}
    n_buses    = length(bus_ids)
    opt_state  = Flux.setup(Flux.Adam(lr), ffnn)
    epoch_losses = Float64[]

    for epoch in 1:n_epochs
        total_loss  = 0.0
        n_valid     = 0

        for point in training_data
            isempty(point.x_opt) && continue   # skip failed solves

            ξ_vec   = encode_ξ(point)
            pg_true = Float64[point.x_opt["pg"][g] for g in gen_ids]

            loss_val, grads = Flux.withgradient(ffnn) do nn
                pg_pred = surrogate_forward(ξ_vec, nn, file_name, n_buses)
                sum((pg_pred .- pg_true) .^ 2)
            end

            Flux.update!(opt_state, ffnn, grads[1])
            total_loss += loss_val
            n_valid    += 1
        end

        mean_loss = n_valid > 0 ? total_loss / n_valid : NaN
        push!(epoch_losses, mean_loss)
        println("Epoch $epoch/$n_epochs: mean loss = $mean_loss  ($n_valid samples)")
    end

    return epoch_losses
end

end
