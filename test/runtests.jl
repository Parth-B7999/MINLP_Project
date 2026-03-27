using Test

# Requires a valid Gurobi license.
# Run from the repo root:  julia --project test/runtests.jl

@testset "MINLP Project" begin
    include("test_formulation.jl")
end
