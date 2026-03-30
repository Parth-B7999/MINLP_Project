"""
casestudy14.jl — Functional tests for surrogates.jl on IEEE case14.

Covers, in order:
  1. gather_training_data    — solve one instance, check status and x_opt structure
  2. encode_ξ                — flat input vector shape and value consistency
  3. encode_x                — flat output vector shape and sanity
  4. save_repertoire /
     load_repertoire         — JLD2 roundtrip
  5. sample_training_data    — small LHS batch (3 samples), solve-rate check
  6. surrogate_output_dim    — formula correctness
  7. build_ffnn              — network construction and forward pass shape
  8. opf_layer + rrule       — Step 2 pass-through on one sample

Run from the repo root:
    julia --project examples/case14/casestudy14.jl
"""

include(joinpath(@__DIR__, "../../src/surrogates.jl"))
using .surrogates
using Test
using Random
import ChainRulesCore

# ── constants ────────────────────────────────────────────────────────────────

const FILE_PATH = joinpath(@__DIR__, "../../data/case14.m")
const TMP_SAVE  = joinpath(@__DIR__, "labeled_data/test_roundtrip.jld2")

# case14 network dimensions (verified via PowerModels)
const N_BUSES    = 14
const N_GEN      = 5
const N_BRANCHES = 20   # also = n_edges for c_ij, s_ij, etc.

# Flat-start linearization point: vr = 1.0, vi = 0.0 for all buses
const FLAT_VR = Dict(string(i) => 1.0 for i in 1:N_BUSES)
const FLAT_VI = Dict(string(i) => 0.0 for i in 1:N_BUSES)


# ── helpers ──────────────────────────────────────────────────────────────────

function check(cond::Bool, msg::String)
    if cond
        println("  PASS  $msg")
    else
        println("  FAIL  $msg")
    end
    cond
end

function section(title::String)
    println("\n" * "="^60)
    println("  $title")
    println("="^60)
end


# ═══════════════════════════════════════════════════════════════
# 1. gather_training_data
# ═══════════════════════════════════════════════════════════════
section("1. gather_training_data")

pt = surrogates.gather_training_data(FILE_PATH, FLAT_VR, FLAT_VI)

@testset "gather_training_data" begin
    @test pt.status in ("OPTIMAL", "LOCALLY_SOLVED")
    println("  Solver status: $(pt.status)")

    # x_opt must contain all expected variable groups
    for key in ("vr", "vi", "c_ii", "c_ij", "s_ij",
                "u", "pg", "qg",
                "p_fr", "q_fr", "p_to", "q_to",
                "xi_c", "xij_c", "xij_s")
        @test haskey(pt.x_opt, key)
    end

    # per-bus variables: 14 entries each
    for key in ("vr", "vi", "c_ii", "xi_c")
        @test length(pt.x_opt[key]) == N_BUSES
    end

    # per-gen variables: 5 entries each
    for key in ("u", "pg", "qg")
        @test length(pt.x_opt[key]) == N_GEN
    end

    # per-branch variables: 20 entries each
    for key in ("p_fr", "q_fr", "p_to", "q_to")
        @test length(pt.x_opt[key]) == N_BRANCHES
    end

    # per-edge variables: 20 entries each
    for key in ("c_ij", "s_ij", "xij_c", "xij_s")
        @test length(pt.x_opt[key]) == N_BRANCHES
    end

    # node_vr / node_vi must match what we passed in
    @test pt.node_vr == FLAT_VR
    @test pt.node_vi == FLAT_VI
end


# ═══════════════════════════════════════════════════════════════
# 2. encode_ξ
# ═══════════════════════════════════════════════════════════════
section("2. encode_ξ")

xi_vec = surrogates.encode_ξ(pt)

@testset "encode_ξ" begin
    @test xi_vec isa Vector{Float32}
    @test length(xi_vec) == 2 * N_BUSES          # 28

    # First N_BUSES entries are vr (all 1.0 at flat start)
    @test all(xi_vec[1:N_BUSES] .≈ 1.0f0)
    # Last N_BUSES entries are vi (all 0.0 at flat start)
    @test all(xi_vec[N_BUSES+1:end] .≈ 0.0f0)

    # No NaN / Inf
    @test all(isfinite.(xi_vec))
end
println("  encode_ξ length: $(length(xi_vec))  (expected $(2*N_BUSES))")


# ═══════════════════════════════════════════════════════════════
# 3. encode_x
# ═══════════════════════════════════════════════════════════════
section("3. encode_x")

x_vec = surrogates.encode_x(pt)

# Expected flat dimension, following encode_x layout order:
#   vr(14) + vi(14) + c_ii(14)                    → 3 bus groups
#   c_ij(20) + s_ij(20)                           → 2 edge groups
#   u(5) + pg(5) + qg(5)                          → 3 gen groups
#   p_fr(20) + q_fr(20) + p_to(20) + q_to(20)    → 4 branch groups
#   xi_c(14)                                       → 1 bus group
#   xij_c(20) + xij_s(20)                         → 2 edge groups
#   total = 4*14 + 4*20 + 3*5 + 4*20 = 56+80+15+80 = 231
const X_DIM = 4*N_BUSES + (2 + 2)*N_BRANCHES + 3*N_GEN + 4*N_BRANCHES
# = 4*14 + 4*20 + 3*5 + 4*20 = 231

# Index of u block in the flat x vector (0-based: after vr+vi+c_ii+c_ij+s_ij)
const U_START = 3*N_BUSES + 2*N_BRANCHES + 1   # = 3*14 + 2*20 + 1 = 83

@testset "encode_x" begin
    @test x_vec isa Vector{Float32}
    @test length(x_vec) == X_DIM
    @test all(isfinite.(x_vec))
    # u ∈ {0,1} — check within [−ε, 1+ε]
    u_vals = x_vec[U_START : U_START + N_GEN - 1]
    @test all(-0.01f0 .<= u_vals .<= 1.01f0)
end
println("  encode_x length: $(length(x_vec))  (expected $X_DIM)")


# ═══════════════════════════════════════════════════════════════
# 4. save_repertoire / load_repertoire
# ═══════════════════════════════════════════════════════════════
section("4. save_repertoire / load_repertoire")

mkpath(dirname(TMP_SAVE))

@testset "save/load repertoire" begin
    pts_orig = [pt, pt]   # reuse the point we already have

    surrogates.save_repertoire(pts_orig, TMP_SAVE)
    @test isfile(TMP_SAVE)

    pts_loaded = surrogates.load_repertoire(TMP_SAVE)
    @test length(pts_loaded) == 2
    @test pts_loaded[1].status == pt.status
    @test pts_loaded[1].node_vr == pt.node_vr
    @test pts_loaded[1].node_vi == pt.node_vi

    # Append 1 more point — total should be 3
    surrogates.save_repertoire([pt], TMP_SAVE)
    pts_appended = surrogates.load_repertoire(TMP_SAVE)
    @test length(pts_appended) == 3

    rm(TMP_SAVE)   # clean up
end


# ═══════════════════════════════════════════════════════════════
# 5. sample_training_data  (small LHS batch)
# ═══════════════════════════════════════════════════════════════
section("5. sample_training_data  (n=3 LHS samples)")

Random.seed!(42)
lhs_points = surrogates.sample_training_data(FILE_PATH, 3)

@testset "sample_training_data" begin
    @test length(lhs_points) == 3
    @test all(p -> p isa surrogates.TrainingDataPoint, lhs_points)

    n_ok = count(p -> p.status in ("OPTIMAL", "LOCALLY_SOLVED"), lhs_points)
    println("  Solve rate: $n_ok / 3")
    @test n_ok >= 1   # at least one must solve

    # Each successful point must have properly shaped x_opt
    for p in filter(p -> !isempty(p.x_opt), lhs_points)
        @test length(p.x_opt["vr"]) == N_BUSES
        @test length(p.x_opt["pg"]) == N_GEN
        @test length(p.x_opt["p_fr"]) == N_BRANCHES
    end
end


# ═══════════════════════════════════════════════════════════════
# 6. surrogate_output_dim
# ═══════════════════════════════════════════════════════════════
section("6. surrogate_output_dim")

@testset "surrogate_output_dim" begin
    # Formula: 2*n_buses + K*(2*n_gen + 1)
    for K in [1, 3, 5]
        expected = 2*N_BUSES + K*(2*N_GEN + 1)
        got = surrogates.surrogate_output_dim(N_BUSES, N_GEN, K)
        @test got == expected
        println("  K=$K  →  output_dim=$got  (expected $expected)")
    end
    # K=0: should collapse to just the linearization point
    @test surrogates.surrogate_output_dim(N_BUSES, N_GEN, 0) == 2*N_BUSES
end


# ═══════════════════════════════════════════════════════════════
# 7. build_ffnn  — construction and forward pass
# ═══════════════════════════════════════════════════════════════
section("7. build_ffnn")

K          = 3
input_dim  = 2 * N_BUSES                           # 28
output_dim = surrogates.surrogate_output_dim(N_BUSES, N_GEN, K)  # 61

nn = surrogates.build_ffnn(input_dim, output_dim, [64, 64])

@testset "build_ffnn" begin
    # Verify architecture depth: 2 hidden + 1 output = 3 Dense layers
    @test length(nn.layers) == 3

    # Forward pass on a random input
    x_in  = randn(Float32, input_dim)
    y_out = nn(x_in)
    @test y_out isa Vector{Float32}
    @test length(y_out) == output_dim
    @test all(isfinite.(y_out))
    println("  Input dim: $input_dim  →  Output dim: $(length(y_out))  (expected $output_dim)")

    # Forward pass using encode_ξ from a real training point
    y_real = nn(xi_vec)
    @test length(y_real) == output_dim
    @test all(isfinite.(y_real))
    println("  Forward pass on encoded ξ: OK")

    # Vary hidden_dims: single hidden layer
    nn_shallow = surrogates.build_ffnn(input_dim, output_dim, [32])
    @test length(nn_shallow.layers) == 2
    @test length(nn_shallow(x_in)) == output_dim

    # No hidden layers: direct linear map
    nn_linear = surrogates.build_ffnn(input_dim, output_dim, Int[])
    @test length(nn_linear.layers) == 1
    @test length(nn_linear(x_in)) == output_dim
end


# ═══════════════════════════════════════════════════════════════
# 8. opf_layer + rrule  (Step 2 pass-through on one sample)
# ═══════════════════════════════════════════════════════════════
section("8. opf_layer + rrule  (Step 2 pass-through)")

# Reuse `pt` from section 1 (flat-start, already OPTIMAL).
# Extract θ̂ in bus-sorted order — the same convention build_diff_opf expects.
bus_ids_sorted = sort(collect(keys(pt.node_vr)), by = k -> parse(Int, k))
θ̂_vr = Float64[pt.node_vr[i] for i in bus_ids_sorted]
θ̂_vi = Float64[pt.node_vi[i] for i in bus_ids_sorted]

println("  Linearization point (first 4 buses):")
println("    θ̂_vr = $(round.(θ̂_vr[1:4], digits=4))")
println("    θ̂_vi = $(round.(θ̂_vi[1:4], digits=4))")

# ── 8a. Forward pass — no learned cuts ──────────────────────────────────────
pg_star = surrogates.opf_layer(FILE_PATH, θ̂_vr, θ̂_vi, nothing, nothing, nothing)

@testset "opf_layer forward (no cuts)" begin
    @test pg_star isa Vector{Float64}
    @test length(pg_star) == N_GEN
    @test all(isfinite.(pg_star))
    @test all(pg_star .>= -1e-6)    # pg ≥ 0 by construction
end
println("  pg* (no cuts): $(round.(pg_star, digits=4))")

# ── 8b. rrule pullback — no learned cuts ────────────────────────────────────
dL_dpg = ones(Float64, N_GEN)   # uniform upstream gradient

pg_fwd, pullback = ChainRulesCore.rrule(
    surrogates.opf_layer, FILE_PATH, θ̂_vr, θ̂_vi, nothing, nothing, nothing
)
(_, _, dθ̂_vr, dθ̂_vi, dα, dβ, dγ) = pullback(dL_dpg)

@testset "rrule pullback (no cuts)" begin
    @test pg_fwd ≈ pg_star atol=1e-6          # forward consistent
    @test dθ̂_vr isa Vector{Float64}
    @test dθ̂_vi isa Vector{Float64}
    @test length(dθ̂_vr) == N_BUSES
    @test length(dθ̂_vi) == N_BUSES
    @test all(isfinite.(dθ̂_vr))
    @test all(isfinite.(dθ̂_vi))
    @test dα isa ChainRulesCore.NoTangent     # no cuts → no cut gradients
    @test dβ isa ChainRulesCore.NoTangent
    @test dγ isa ChainRulesCore.NoTangent
end
println("  dθ̂_vr (first 4): $(round.(dθ̂_vr[1:4], digits=6))")
println("  dθ̂_vi (first 4): $(round.(dθ̂_vi[1:4], digits=6))")

# ── 8c. Forward pass — with learned cuts ────────────────────────────────────
# Use small α, β so the cuts barely constrain the feasible set, and a loose γ
# so the problem stays feasible.
K_cuts = 3
Random.seed!(7)
α_test = 0.001 * randn(K_cuts, N_GEN)
β_test = 0.001 * randn(K_cuts, N_GEN)
γ_test = fill(1e4, K_cuts)   # non-binding upper bound

pg_star_cuts = opf_layer(FILE_PATH, θ̂_vr, θ̂_vi, α_test, β_test, γ_test)

@testset "opf_layer forward (with cuts)" begin
    @test pg_star_cuts isa Vector{Float64}
    @test length(pg_star_cuts) == N_GEN
    @test all(isfinite.(pg_star_cuts))
end
println("  pg* (with cuts): $(round.(pg_star_cuts, digits=4))")

# ── 8d. rrule pullback — with learned cuts ───────────────────────────────────
pg_fwd_c, pullback_c = ChainRulesCore.rrule(
    opf_layer, FILE_PATH, θ̂_vr, θ̂_vi, α_test, β_test, γ_test
)
(_, _, dθ̂_vr_c, dθ̂_vi_c, dα_c, dβ_c, dγ_c) = pullback_c(dL_dpg)

@testset "rrule pullback (with cuts)" begin
    @test pg_fwd_c ≈ pg_star_cuts atol=1e-6
    @test dθ̂_vr_c isa Vector{Float64} && length(dθ̂_vr_c) == N_BUSES
    @test dθ̂_vi_c isa Vector{Float64} && length(dθ̂_vi_c) == N_BUSES
    @test dα_c isa Matrix{Float64}    && size(dα_c) == (K_cuts, N_GEN)
    @test dβ_c isa Matrix{Float64}    && size(dβ_c) == (K_cuts, N_GEN)
    @test dγ_c isa Vector{Float64}    && length(dγ_c) == K_cuts
    @test all(isfinite.(dα_c))
    @test all(isfinite.(dβ_c))
    @test all(isfinite.(dγ_c))
end
println("  dα (K=3, gen 1): $(round.(dα_c[:, 1], digits=6))")
println("  dγ             : $(round.(dγ_c, digits=6))")


# ── summary ──────────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("  All tests complete.")
println("="^60)
