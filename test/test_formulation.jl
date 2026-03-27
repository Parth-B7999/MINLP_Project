using Test
using JuMP
using DiffOpt
using PowerModels

include(joinpath(@__DIR__, "..", "src", "formulation.jl"))

# All build_* functions are called as formulation.build_... because only
# convex_ac_uc, build_convex_ac_uc, build_diff_opf, ac_uc, mp_ac_uc are exported.

# All tests use case14 — smallest available case
const CASE14 = joinpath(@__DIR__, "..", "data", "case14.m")

# Helper: parse bus IDs and build a flat voltage profile (vr=1, vi=0)
function flat_voltage_profile(file_path::String)
    raw = PowerModels.parse_file(file_path)
    bus_ids = collect(keys(raw["bus"]))
    node_vr = Dict(i => 1.0 for i in bus_ids)
    node_vi = Dict(i => 0.0 for i in bus_ids)
    return node_vr, node_vi, bus_ids
end

# ============================================================
# Single-period rectangular AC-UC  (build_single_ac_uc_rectangular)
# ============================================================

@testset "build_single_ac_uc_rectangular" begin
    model = formulation.build_single_ac_uc_rectangular(CASE14)

    @testset "model structure" begin
        @test model isa JuMP.Model
        @test haskey(model, :vr)
        @test haskey(model, :vi)
        @test haskey(model, :pg)
        @test haskey(model, :qg)
        @test haskey(model, :u)
        @test haskey(model, :p_fr)
        @test haskey(model, :q_fr)
        @test haskey(model, :p_to)
        @test haskey(model, :q_to)
    end

    @testset "binary unit commitment" begin
        for var in model[:u]
            @test is_binary(var)
        end
    end

    @testset "solves to optimality" begin
        set_silent(model)
        optimize!(model)
        @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        @test objective_value(model) > 0
    end
end


# ============================================================
# Multi-period rectangular AC-UC  (build_mp_ac_uc_rectangular)
# ============================================================

#= 
@testset "build_mp_ac_uc_rectangular" begin
    demand_curve = [0.8, 1.0, 1.2]
    model = formulation.build_mp_ac_uc_rectangular(CASE14, demand_curve)

    @testset "model structure" begin
        @test model isa JuMP.Model
        @test haskey(model, :vr)
        @test haskey(model, :u)
        @test size(model[:pg], 2) == 3   # variables span all 3 time periods
    end

    @testset "binary unit commitment" begin
        for var in model[:u]
            @test is_binary(var)
        end
    end

    @testset "solves to optimality" begin
        set_silent(model)
        optimize!(model)
        @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    end
end
=#

# ============================================================
# Convex QCQP  (build_convex_ac_uc)
# ============================================================

@testset "build_convex_ac_uc" begin
    node_vr, node_vi, _ = flat_voltage_profile(CASE14)
    model = formulation.build_convex_ac_uc(CASE14, node_vr, node_vi)

    @testset "model structure" begin
        @test model isa JuMP.Model
        @test haskey(model, :vr)
        @test haskey(model, :vi)
        @test haskey(model, :c_ii)
        @test haskey(model, :c_ij)
        @test haskey(model, :s_ij)
        @test haskey(model, :u)
        @test haskey(model, :pg)
        @test haskey(model, :xi_c)
        @test haskey(model, :xij_c)
        @test haskey(model, :xij_s)
    end

    @testset "binary unit commitment retained" begin
        for var in model[:u]
            @test is_binary(var)
        end
    end

    @testset "solves to optimality" begin
        set_silent(model)
        optimize!(model)
        @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)

        # convex relaxation slacks must be non-negative at solution
        t = 1
        for i in axes(model[:xi_c], 1)
            @test value(model[:xi_c][i, t]) >= -1e-6
        end
    end
end


# ============================================================
# Differentiable OPF layer  (build_diff_opf)
# ============================================================

@testset "build_diff_opf" begin
    node_vr, node_vi, bus_ids = flat_voltage_profile(CASE14)
    n = length(bus_ids)
    θ̂_vr = ones(Float64, n)
    θ̂_vi = zeros(Float64, n)

    model = formulation.build_diff_opf(CASE14, θ̂_vr, θ̂_vi)

    @testset "model structure" begin
        @test model isa JuMP.Model
        @test haskey(model, :vr)
        @test haskey(model, :vi)
        @test haskey(model, :c_ii)
        @test haskey(model, :pg)
        @test haskey(model, :xi_c)
        @test haskey(model, :node_vr)
        @test haskey(model, :node_vi)
    end

    @testset "UC variables excluded" begin
        @test !haskey(model, :u)
    end

    @testset "node_vr and node_vi have correct dimensions" begin
        @test length(model[:node_vr]) == n
        @test length(model[:node_vi]) == n
    end

    @testset "solves to optimality" begin
        set_silent(model)
        optimize!(model)
        @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)

        # pg must lie within [pmin, pmax] — continuous bounds, not UC-coupled
        raw = PowerModels.parse_file(CASE14)
        t = 1
        for (i, gen) in raw["gen"]
            pg_val = value(model[:pg][i, t])
            @test pg_val >= gen["pmin"] - 1e-6
            @test pg_val <= gen["pmax"] + 1e-6
        end
    end

    @testset "dimension assertion fires on bad input" begin
        @test_throws AssertionError formulation.build_diff_opf(CASE14, ones(n+1), zeros(n))
    end
end
