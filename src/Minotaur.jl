#
# Wrapper for the serial Minotaur interface
#

module Minotaur

function convert_to_c_idx(indicies)
    for i in 1:length(indicies)
        indicies[i] = indicies[i] - 1
    end
end

type MinotaurProblem
    ref::Ptr{Void} # reference to the internal data structure
    n::Int # number of variables
    m::Int # number of constraints
    x::Vector{Float64} # starting and final solution
    g::Vector{Float64} # final constraint valaue
    obj_val::Float64 # final objective
    status::Int # final status

    # Callbacks
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h  # Can be nothing
    
    #jac , hess
    nzJac::Int
    nzHess::Int

    # For MathProgBase
    sense::Symbol

    function MinotaurProblem(
        ref::Ptr{Void}, n, m, 
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, nzJac, nzHess)
        prob = new(ref, n, m, zeros(Float64, n), zeros(Float64, m), 0.0, 0,
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, nzJac, nzHess,
        :Min)
        # Free the internal MinotaurProblem structure when
        # the Julia MinotaurProblem instance goes out of scope
        finalizer(prob, freeProblem)
        # Return the object we just made
        prob
    end
end

ApplicationReturnStatus = Dict(
0=>:Solve_Succeeded)

###########################################################################
# Callback wrappers
###########################################################################
# Objective (eval_f)
function eval_f_wrapper(x_ptr::Ptr{Float64}, obj_ptr::Ptr{Float64}, user_data::Ptr{Void})
    # println(" julia - eval_f_wrapper " ); 
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::MinotaurProblem
    # Calculate the new objective
    new_obj = convert(Float64, prob.eval_f(pointer_to_array(x_ptr, prob.n)))::Float64
    # Fill out the pointer
    unsafe_store!(obj_ptr, new_obj)
    # Done
    return Int32(1)
end

# Constraints (eval_g)
function eval_g_wrapper(x_ptr::Ptr{Float64}, g_ptr::Ptr{Float64}, user_data::Ptr{Void})
    # println(" julia - eval_g_wrapper " ); 
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::MinotaurProblem
    # Calculate the new constraint values
    new_g = pointer_to_array(g_ptr, prob.m)
    prob.eval_g(pointer_to_array(x_ptr, prob.n), new_g)
    # Done
    return Int32(1)
end

# Objective gradient (eval_grad_f)
function eval_grad_f_wrapper(x_ptr::Ptr{Float64}, grad_f_ptr::Ptr{Float64}, user_data::Ptr{Void})
    # println(" julia -  eval_grad_f_wrapper " );    
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::MinotaurProblem
    # Calculate the gradient
    new_grad_f = pointer_to_array(grad_f_ptr, Int(prob.n))
    prob.eval_grad_f(pointer_to_array(x_ptr, Int(prob.n)), new_grad_f)
    if prob.sense == :Max
        new_grad_f *= -1.0
    end
    # Done
    return Int32(1)
end

# Jacobian (eval_jac_g)
function eval_jac_g_wrapper(x_ptr::Ptr{Float64}, values_ptr::Ptr{Float64}, iRow::Ptr{Cint}, jCol::Ptr{Cint},  user_data::Ptr{Void})
    # println(" julia -  eval_jac_g_wrapper " );
    # Extract Julia the problem from the pointer  
    #@show user_data  
    prob = unsafe_pointer_to_objref(user_data)::MinotaurProblem
    #@show prob
    # Determine mode
    mode = (values_ptr == C_NULL) ? (:Structure) : (:Values)
    x = pointer_to_array(x_ptr, prob.n)
    irows = pointer_to_array(iRow, Int(prob.nzJac))
    kcols = pointer_to_array(jCol, Int(prob.n+1))
    values = pointer_to_array(values_ptr, Int(prob.nzJac))
    prob.eval_jac_g(x, mode, irows, kcols, values)
    if mode == :Structure 
        convert_to_c_idx(irows)
        convert_to_c_idx(kcols)
    end
    # Done
    return Int32(1)
end

# Hessian
function eval_h_wrapper(x_ptr::Ptr{Float64}, lambda_ptr::Ptr{Float64}, values_ptr::Ptr{Float64}, iRow::Ptr{Cint}, jCol::Ptr{Cint}, user_data::Ptr{Void})
    # println(" julia - eval_h_wrapper " ); 
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::MinotaurProblem
    # Did the user specify a Hessian
    if prob.eval_h === nothing
        # No Hessian provided
        return Int32(0)
    else
        # Determine mode
        mode = (values_ptr == C_NULL) ? (:Structure) : (:Values)
        x = pointer_to_array(x_ptr, prob.n)
        lambda = pointer_to_array(lambda_ptr, prob.m)
        irows = pointer_to_array(iRow, Int(prob.nzHess))
        kcols = pointer_to_array(jCol, Int(prob.n+1))
        values = pointer_to_array(values_ptr, Int(prob.nzHess))
        obj_factor = 1.0
        if prob.sense == :Max
            obj_factor *= -1.0
        end
        prob.eval_h(x, mode, irows, kcols, obj_factor, lambda, values)
        if mode == :Structure
            convert_to_c_idx(irows)
            convert_to_c_idx(kcols)
        end
        # Done
        return Int32(1)
    end
end

###########################################################################
# C function wrappers
###########################################################################
function createProblem(n::Int, m::Int,
    x_L::Vector{Float64}, x_U::Vector{Float64},
    g_L::Vector{Float64}, g_U::Vector{Float64},
    nzJac::Int, nzHess::Int,
    objSense::Int, nonlinObj::Bool, numObj::Int, 
    eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h = nothing)

    @assert n == length(x_L) == length(x_U)
    @assert m == length(g_L) == length(g_U)
    eval_f_cb = cfunction(eval_f_wrapper,Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Void}) )
    eval_g_cb = cfunction(eval_g_wrapper,Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Void}) )
    eval_grad_f_cb = cfunction(eval_grad_f_wrapper, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Void}) )
    eval_jac_g_cb = cfunction(eval_jac_g_wrapper, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}, Ptr{Void}))
    eval_h_cb = cfunction(eval_h_wrapper, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}, Ptr{Void}))
    
    # create Minotaur Environment API  
    env = ccall((:createEnv, "libminotaur_shared"),  Ptr{Void}, ())

    # load problem parameters to Julia Interface  
    ccall((:loadJuliaInterface, "libminotaur_shared"), Void, (Ptr{Void}, Cint, Cint, 
    Ptr{Float64}, Ptr{Float64},
    Ptr{Float64}, Ptr{Float64}, 
    Cint, Cint, 
    Cint, Cint), 
    env, n, m, 
    x_L, x_U, 
    g_L, g_U , 
    nzJac, nzHess, 
    obj_sense, 
    is_nl_obj, nb_obj)

    # set callback functions 
    ccall((:setCallbacks, "libminotaur_shared"), Void, 
                                                (Ptr{Void}, Ptr{Void}, Ptr{Void},
                                                 Ptr{Void}, Ptr{Void}, Ptr{Void}), 
                                                 env, eval_f_cb, eval_g_cb, 
                                                 eval_grad_f_cb, eval_jac_g_cb, eval_h_cb)

    #if ret == C_NULL
        #error("Minotaur: Failed to construct problem.")
    #else
        #return(MinotaurProblem(ret, n, m, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, nzJac, nzHess))
    #end
end

function solveProblem(prob::MinotaurProblem)
    # @show "solveProblem"    
    
    final_objval = [0.0]
    ret = ccall((:solveProblem,"libminotaur_shared"), Cint, 
            (Ptr{Void}, Ptr{Float64}, Ptr{Float64}, Any),
            prob.ref, final_objval, prob.x, prob)
    prob.obj_val = final_objval[1]
    prob.status = Int(ret)

    return prob.status
end

function freeProblem(prob::MinotaurProblem)
    # @show "freeProblem"
    ret = ccall((:freeProblem, "libminotaur_shared"),
            Void, (Ptr{Void},),
            prob.ref)
    # @show ret
    return ret
end

include("MinotaurSolverInterface.jl")

end # module