include("formulation.jl")
include("utils.jl")

using .formulation
using JuMP
using DataFrames
using ProgressMeter

"""
Case study: Unit commitment with AC OPF

Using convex QC approximation for ACOPF:
https://link.springer.com/article/10.1007/s11081-026-10079-4
"""


"""
    solve_convex_MINLP(file_path, parameter_sets)

Solve multiple instances of the convex AC OPF+UC problem and return a results DataFrame.

Each element of `parameter_sets` is a `(node_vr, node_vi)` tuple of Dicts keyed by bus id.
Because `node_vr` and `node_vi` are baked into constraint coefficients (not JuMP variables),
a fresh model is built for each parameter set.

Returns a DataFrame with columns: `node_vr`, `node_vi`, `x_opt`, `status`.
"""
function solve_convex_MINLP(file_path::String, parameter_sets::Vector{Tuple{Dict,Dict}})
    results = DataFrame(
        node_vr = Dict[],
        node_vi = Dict[],
        x_opt   = Dict[],
        status  = String[],
    )

    p_bar = Progress(length(parameter_sets); dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:cyan)

    for (node_vr, node_vi) in parameter_sets
        try
            model = build_convex_ac_uc(file_path, node_vr, node_vi)
            optimize!(model)

            stat = string(termination_status(model))

            if stat in ("OPTIMAL", "LOCALLY_SOLVED")
                t = 1
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
                push!(results, (node_vr=node_vr, node_vi=node_vi, x_opt=x_opt, status=stat))
            else
                push!(results, (node_vr=node_vr, node_vi=node_vi, x_opt=Dict(), status=stat))
            end

        catch e
            push!(results, (node_vr=node_vr, node_vi=node_vi, x_opt=Dict(), status="ERROR: $(typeof(e))"))
        finally
            next!(p_bar)
        end
    end

    return results
end
