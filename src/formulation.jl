# Optimization Problem Structure
using JuMP, DiffOpt, Ipopt, Gurobi

# Surrogate Model Architecture
using Flux

"""
Here is where we will maintain the problem structure and the core functions to running this problem.
"""


struct TrainingData
    u_bar::Vector{Float64}
    x_star::Vector{Float64}
end  


function formulate_monolithic()
    """
    Function to formulate the MIQCQP problem.
    """
end


function solve_monolithic(model::JuMP.Model)
    """
    Function to gather training data for the convex MIQCQP.
        model - JuMP model containing the monolithic MIQCP
    Returns:
        training_data - mapping of the fixed parameters \bar{u} --> x*
    """
    optimize!(model)
    
    u = value.(model[u])
    
    return 
end