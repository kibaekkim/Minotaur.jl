# Standard interface
importall MathProgBase.SolverInterface
import MathProgBase

###############################################################################
# Solver objects
export MinotaurSolver
immutable MinotaurSolver <: AbstractMathProgSolver
    options
end
MinotaurSolver(;kwargs...) = MinotaurSolver(kwargs)

type MinotaurMathProgModel <: AbstractMathProgModel
    internal::MinotaurProblem
    numvar::Int
    numconstr::Int
    numObj::Int # number of objective functions - multi objective 
    
    x_l::Vector{Float64}
    x_u::Vector{Float64}
    g_l::Vector{Float64}
    g_u::Vector{Float64}

    sense::Symbol
    lin_constrs::Array{Dict{Int, Float64}}
    num_linconstr::Int
    num_quadconstr::Int
    num_nonlinconstr::Int
    lin_obj::Dict{Int, Float64}

    varType::Vector{Char}
    nonlinearObj::Bool
    warmstart::Vector{Float64}
    options

    function MinotaurMathProgModel(;options...)
        model = new()
        model.options   = options
        model.numvar    = 0
        model.numconstr = 0
	model.numObj = 0
        model.nonlinearObj = false
        model.x_l = zeros(0)
        model.x_u = zeros(0)
        model.g_l = zeros(0)
        model.g_u = zeros(0)
        model.sense = :Min 
        model.warmstart = Float64[]
	model.lin_constrs = Dict{Int, Float64}[]
        model.num_linconstr = 0
	model.num_quadconstr = 0
        model.num_nonlinconstr = 0
	model.lin_obj = Dict{Int, Float64}()
        model.varType = Char[]
        model
    end
end

type MinotaurLinearQuadraticModel <: AbstractLinearQuadraticModel
    inner::MinotaurMathProgModel
end

type MinotaurNonlinearModel <: AbstractNonlinearModel
    inner::MinotaurMathProgModel
end

NonlinearModel(s::MinotaurSolver) = MinotaurNonlinearModel(
    MinotaurMathProgModel(;s.options...)
)

LinearQuadraticModel(s::MinotaurSolver) = MinotaurLinearQuadraticModel(
    MinotaurMathProgModel(;s.options...)
)

###############################################################################
# Begin interface implementation

# generic nonlinear interface
function loadproblem!(outer::MinotaurNonlinearModel,
                      numVar::Int,
                      numConstr::Int,
                      x_l, x_u, g_lb, g_ub,
                      sense::Symbol,
                      d::AbstractNLPEvaluator)
    m = outer.inner 
    m.numvar = numVar
    m.numconstr = numConstr
    m.nonlinearObj = !isobjlinear(d)   # if objective function is nonlinear, return true 
      
    features = features_available(d)
    has_hessian = (:Hess in features)
    if has_hessian
        initialize(d, [:Grad, :Jac, :Hess])
        Ihess, Jhess = hesslag_structure(d)
    else
        initialize(d, [:Grad, :Jac])
        Ihess = Int[]
        Jhess = Int[]
    end
    Ijac, Jjac = jac_structure(d)
    @assert length(Ijac) == length(Jjac)
    @assert length(Ihess) == length(Jhess)
    @assert sense == :Min || sense == :Max

    # Objective callback
    scl = (sense == :Min) ? 1.0 : -1.0
    eval_f_cb(x) = scl*eval_f(d,x)

    # Objective gradient callback
    eval_grad_f_cb(x, grad_f) = (eval_grad_f(d, grad_f, x); scale!(grad_f,scl))

    # Constraint value callback
    eval_g_cb(x, g) = eval_g(d, g, x)

    # Jacobian callback
    function eval_jac_g_cb(x, mode, rows, cols, values)
        if mode == :Structure
            for i in 1:length(Ijac)
                rows[i] = Ijac[i]
                cols[i] = Jjac[i]
            end
        else
            eval_jac_g(d, values, x)
        end
    end

    if has_hessian
        # Hessian callback
        function eval_h_cb(x, mode, rows, cols, obj_factor,
            lambda, values)
            if mode == :Structure
                for i in 1:length(Ihess)
                    rows[i] = Ihess[i]
                    cols[i] = Jhess[i]
                end
            else
                obj_factor *= scl
                eval_hesslag(d, values, x, obj_factor, lambda)
            end
        end
    else
        eval_h_cb = nothing
    end
    @show numVar 
    @show numConstr
    @show x_l
    @show x_u
    @show g_lb
    @show g_ub
    @show length(Ihess)
    @show sense
    @show m.nonlinearObj
    @show eval_g_cb
    @show m.varType
    m.internal = createProblem(
        numVar, numConstr, float(x_l), float(x_u), 
        float(g_lb), float(g_ub), 
        length(Ijac), length(Ihess), sense, m.nonlinearObj, 
        eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb, eval_h_cb)
    if !has_hessian
        addOption(m.internal, "hessian_approximation", "limited-memory")
    end
end

function loadproblem!(outer::MinotaurLinearQuadraticModel, A::AbstractMatrix, 
			x_l, x_u, c, g_l, g_u, sense::Symbol)
    m = outer.inner # inner is MinotaurMathProgModel
    m.numconstr, m.numvar = size(A)

    @assert m.numvar == length(x_l) == length(x_u)
    @assert m.numconstr == length(g_l) == length(g_u)

    m.x_l, m.x_u = x_l, x_u
    m.g_l, m.g_u = g_l, g_u
    
    # set the sense of objective function
    setsense!(m, sense)
    
    # initiate linear constraints 
    m.lin_constrs = [Dict{Int, Float64}() for _ in 1:m.numconstr]
    
    #load matrix A 
    for var = 1:A.n, k= A.colptr[var] : (A.colptr[var+1]-1)
        m.lin_constrs[A.rowval[k]][var] = A.nzval[k]
     end
  
    m.nonlinearObj = false 
    m.numObj = 1
    # store linear objective function 
    for (index, val) in enumerate(c)
        m.lin_obj[index] = val
    end
    @show m.lin_obj
    # check validity of constraint bounds 
    for j=1:m.numconstr
        lower = m.g_l[j]
	upper = m.g_u[j]
	if lower == -Inf && upper == Inf
	    error("No bounds on constraint $j provided")
	end
    end 
    
    Ijac = Int[]
    Ihess = Int[]
    # number of each type constraints 
    m.num_linconstr = numlinconstr(m) 

    m.internal = createProblem(
        m.numvar, m.numconstr, float(x_l), float(x_u), 
        float(g_l), float(g_u), 
        length(Ijac), length(Ihess), sense, m.nonlinearObj, m.numObj, 
	m.num_linconstr, m.num_quadconstr, m.num_nonlinconstr, m.lin_obj, m.lin_constrs)
  

end

getsense(m::MinotaurMathProgModel) = m.inner.sense

function setsense!(m::MinotaurMathProgModel, sense::Symbol)
    @assert sense == :Min || sense == :Max
    m.sense = sense 
end

numvar(m::MinotaurMathProgModel) = m.numvar
numconstr(m::MinotaurMathProgModel) = m.numconstr

# TODO: We would need to implement numquad,sos, and nonlinear 
numlinconstr(m::MinotaurMathProgModel)=size(m.lin_constrs,1)

#numquadconstr(m::MinotaurMathProgModel) = num_qconstr(m)
#numsosconstr(m::MinotaurMathProgModel) = length(m.sosconstr)
#numnonlinconstr(m::MinotaurMathProgModel) = length(m.nlpdata.nlconstr)

function optimize!(m::MinotaurMathProgModel)
    
    m.internal.status = :NotOptimized
    #= copy!(m.inner.x, m.warmstart) # set warmstart
    for (name,value) in m.options
        sname = string(name)
        if match(r"(^resto_)", sname) != nothing
            sname = replace(sname, r"(^resto_)", "resto.")
        end
        addOption(m.inner, sname, value)
    end=#
    solveProblem(m.internal)
end

function status(m::MinotaurMathProgModel)
    # Map all the possible return codes, as enumerated in
    # Ipopt.ApplicationReturnStatus, to the MPB statuses:
    # :Optimal, :Infeasible, :Unbounded, :UserLimit, and :Error
 
    stat_sym = m.internal.status
    @show stat_sym
    if  stat_sym == :Solve_Succeeded ||
        stat_sym == :Solved_To_Acceptable_Level
        return :Optimal
    elseif stat_sym == :Infeasible_Problem_Detected
        return :Infeasible
    elseif stat_sym == :Diverging_Iterates
        return :Unbounded
        # Things that are more likely to be fixable by changing
        # a parameter will be treated as UserLimit, although
        # some are error-like too.
    elseif stat_sym == :User_Requested_Stop ||
        stat_sym == :Maximum_Iterations_Exceeded ||
        stat_sym == :Maximum_CpuTime_Exceeded
        return :UserLimit
    else
        # Default is to not mislead user that it worked
        # Includes:
        #   :Search_Direction_Becomes_Too_Small
        #   :Feasible_Point_Found
        #   :Restoration_Failed
        #   :Error_In_Step_Computation
        #   :Not_Enough_Degrees_Of_Freedom
        #   :Invalid_Problem_Definition
        #   :Invalid_Option
        #   :Invalid_Number_Detected
        #   :Unrecoverable_Exception
        #   :NonIpopt_Exception_Thrown
        #   :Insufficient_Memory
        #   :Internal_Error
        warn("Minotaur finished with status $stat_sym")
        return :Error
    end

end
getobjval(m::MinotaurMathProgModel) = m.inner.obj_val * (m.inner.sense == :Max ? -1 : +1)
getsolution(m::MinotaurMathProgModel) = m.inner.x

function getreducedcosts(m::MinotaurMathProgModel)
    sense = m.inner.sense
    redcost = m.inner.mult_x_U - m.inner.mult_x_L
    return sense == :Max ? redcost : -redcost
end
function getconstrduals(m::MinotaurMathProgModel)
    v = m.inner.mult_g # return multipliers for all constraints
    return m.inner.sense == :Max ? copy(v) : -v
end

getrawsolver(m::MinotaurMathProgModel) = m.inner
setwarmstart!(m::MinotaurMathProgModel, x) = (m.warmstart = x)

const rev_var_type_map = Dict(
    :Cont => 'C', 
    :Bin => 'B',
    :Int => 'I'
)

function setvartype!(m::MinotaurMathProgModel,typ::Vector{Symbol})
    @assert all(x->(x in [:Cont, :Bin, :Int]), typ)
    m.varType = map(t->rev_var_type_map[t], typ)
end

freemodel!(m::MinotaurMathProgModel) = freeProblem(m.inner)

# Wrapper functions 
for f in [:optimize!,:status]
    @eval $f(m::MinotaurNonlinearModel) = $f(m.inner)
    @eval $f(m::MinotaurLinearQuadraticModel) = $f(m.inner)
end

for f in [:setvartype!]
    @eval $f(m::MinotaurNonlinearModel, x) = $f(m.inner, x)
    @eval $f(m::MinotaurLinearQuadraticModel,x) = $f(m.inner, x)
end
