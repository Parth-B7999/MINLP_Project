"""
gen_dataset.jl — Generate LHS training data for IEEE case14 UCACOPF.

Samples N_SAMPLES linearization points via Latin Hypercube Sampling over
per-bus voltage bounds (vmin, vmax from case14.m), solves the convex
AC OPF+UC for each, and saves the results to SAVE_PATH.

Run from the repo root:
    julia --project examples/case14/gen_dataset.jl
"""

include(joinpath(@__DIR__, "../../src/surrogates.jl"))
using .surrogates
using Random

const FILE_PATH  = joinpath(@__DIR__, "../../data/case14.m")
const SAVE_PATH  = joinpath(@__DIR__, "labeled_data/repertoire.jld2")
const N_SAMPLES  = 100
const SEED       = 42

Random.seed!(SEED)

println("Generating $N_SAMPLES UCACOPF instances (case14, LHS sampling)...")
mkpath(dirname(SAVE_PATH))

points = sample_training_data(FILE_PATH, N_SAMPLES; save_path=SAVE_PATH)

n_ok   = count(p -> p.status in ("OPTIMAL", "LOCALLY_SOLVED"), points)
n_fail = N_SAMPLES - n_ok
println("\nSolve rate : $n_ok / $N_SAMPLES  ($(round(100*n_ok/N_SAMPLES, digits=1))%)")
println("Failures   : $n_fail")
println("Saved to   : $SAVE_PATH")
