module Hiop
using Libdl
using LinearAlgebra

export createProblem, addOption
export openOutputFile, setProblemScaling, setIntermediateCallback
export solveProblem
export HiopProblem

function __init__()
    try
        path_to_lib = ENV["JULIA_HIOP_LIBRARY_PATH"]
        # Libdl.dlopen(path_to_lib * "/lib/libhiop.so", Libdl.RTLD_GLOBAL)
        Libdl.dlopen("libf77blas.so", Libdl.RTLD_GLOBAL)
        Libdl.dlopen("liblapack.so", Libdl.RTLD_GLOBAL)
        Libdl.dlopen("libhiop.so", Libdl.RTLD_GLOBAL)
        Libdl.dlopen("libchiopInterface.so", Libdl.RTLD_GLOBAL)
    catch
        @warn("Could not load HiOp shared library. Make sure the ENV variable 'JULIA_HIOP_LIBRARY_PATH' points to its location.")
        rethrow()
    end
end

mutable struct cHiopProblem
    refcppHiop::Ptr{Cvoid}
    jprob::Ptr{Cvoid}
    get_prob_sizes::Ptr{Cvoid}
    get_vars_info::Ptr{Cvoid}
    get_cons_info::Ptr{Cvoid}
    eval_f::Ptr{Cvoid}
    eval_gradf::Ptr{Cvoid}
    eval_cons::Ptr{Cvoid}
    get_sparse_dense_blocks_info::Ptr{Cvoid}
    eval_Jac_cons::Ptr{Cvoid}
    eval_Hess_Lagr::Ptr{Cvoid}
    function cHiopProblem()
        return new(C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL)
    end
end

mutable struct HiopProblem
    cprob::cHiopProblem  # Reference to the C data structure
    n::Int64  # Num vars
    m::Int64  # Num cons
    nd::Int64 # Dense whatever
    ns::Int64 # Sparse whatever
    x::Vector{Float64}  # Starting and final solution
    g::Vector{Float64}  # Final constraint values
    mult_g::Vector{Float64} # lagrange multipliers on constraints
    mult_x_L::Vector{Float64} # lagrange multipliers on lower bounds
    mult_x_U::Vector{Float64} # lagrange multipliers on upper bounds
    obj_val::Float64  # Final objective
    status::Int  # Final status

    # Callbacks
    eval_f::Function
    eval_cons::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h  # Can be nothing
    user_data::Any 

    function HiopProblem(
        n, m, nd, ns,
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, user_data)
        prob = new(cHiopProblem(), n, m, nd, ns, zeros(Float64, n), zeros(Float64, m), zeros(Float64,m),
                   zeros(Float64,n), zeros(Float64,n), 0.0, 0,
                   eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, user_data)
        prob.cprob.jprob = pointer_from_objref(prob)
        # Free the internal HiopProblem structure when
        # the Julia HiopProblem instance goes out of scope
        finalizer(freeProblem, prob)
        return prob
    end
end

# From Ipopt/src/Interfaces/IpReturnCodes_inc.h
ApplicationReturnStatus = Dict(
0=>:Solve_Succeeded,
1=>:Solved_To_Acceptable_Level,
2=>:Infeasible_Problem_Detected,
3=>:Search_Direction_Becomes_Too_Small,
4=>:Diverging_Iterates,
5=>:User_Requested_Stop,
6=>:Feasible_Point_Found,
-1=>:Maximum_Iterations_Exceeded,
-2=>:Restoration_Failed,
-3=>:Error_In_Step_Computation,
-4=>:Maximum_CpuTime_Exceeded,
-10=>:Not_Enough_Degrees_Of_Freedom,
-11=>:Invalid_Problem_Definition,
-12=>:Invalid_Option,
-13=>:Invalid_Number_Detected,
-100=>:Unrecoverable_Exception,
-101=>:NonIpopt_Exception_Thrown,
-102=>:Insufficient_Memory,
-199=>:Internal_Error)


###########################################################################
# Callback wrappers
###########################################################################
# Objective (eval_f)
function eval_f_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, obj_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::HiopProblem
    # Calculate the new objective
    new_obj = convert(Float64, prob.eval_f(unsafe_wrap(Array,x_ptr, Int(n)), prob))::Float64
    # Fill out the pointer
    unsafe_store!(obj_ptr, new_obj)
    # Done
    return Int32(1)
end

# Constraints (eval_g)
function eval_g_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, m::Cint, g_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::HiopProblem
    # Calculate the new constraint values
    new_g = unsafe_wrap(Array,g_ptr, Int(m))
    prob.eval_g(unsafe_wrap(Array,x_ptr, Int(n)), new_g)
    # Done
    return Int32(1)
end

# Objective gradient (eval_grad_f)
function eval_grad_f_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, grad_f_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::HiopProblem
    # Calculate the gradient
    new_grad_f = unsafe_wrap(Array,grad_f_ptr, Int(n))
    prob.eval_grad_f(unsafe_wrap(Array,x_ptr, Int(n)), new_grad_f)
    # Done
    return Int32(1)
end

# Jacobian (eval_jac_g)
function eval_jac_g_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, m::Cint, nele_jac::Cint, iRow::Ptr{Cint}, jCol::Ptr{Cint}, values_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::HiopProblem
    # Determine mode
    mode = (values_ptr == C_NULL) ? (:Structure) : (:Values)
    x = unsafe_wrap(Array, x_ptr, Int(n))
    rows = unsafe_wrap(Array, iRow, Int(nele_jac))
    cols = unsafe_wrap(Array,jCol, Int(nele_jac))
    values = unsafe_wrap(Array,values_ptr, Int(nele_jac))
    prob.eval_jac_g(x, mode, rows, cols, values)
    # Done
    return Int32(1)
end

# Hessian
function eval_h_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, obj_factor::Float64, m::Cint, lambda_ptr::Ptr{Float64}, new_lambda::Cint, nele_hess::Cint, iRow::Ptr{Cint}, jCol::Ptr{Cint}, values_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::HiopProblem
    # Did the user specify a Hessian
    if prob.eval_h === nothing
        # No Hessian provided
        return Int32(0)
    else
        # Determine mode
        mode = (values_ptr == C_NULL) ? (:Structure) : (:Values)
        x = unsafe_wrap(Array,x_ptr, Int(n))
        lambda = unsafe_wrap(Array,lambda_ptr, Int(m))
        rows = unsafe_wrap(Array,iRow, Int(nele_hess))
        cols = unsafe_wrap(Array,jCol, Int(nele_hess))
        values = unsafe_wrap(Array,values_ptr, Int(nele_hess))
        prob.eval_h(x, mode, rows, cols, obj_factor, lambda, values)
        # Done
        return Int32(1)
    end
end

###########################################################################
# C function wrappers
###########################################################################
function createProblem(ns::Int, n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
    m::Int, g_L::Vector{Float64}, g_U::Vector{Float64},
    nele_jac::Int, nele_hess::Int,
    eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h = nothing, user_data = nothing)
    # Wrap callbacks
    prob = HiopProblem(n, m, ns, ns, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, user_data)
    prob.cprob.eval_f = @cfunction(eval_f_wrapper, Cint,
                    (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Cvoid}))
    ret = ccall(:hiop_createProblem, Cint, 
    (Ptr{cHiopProblem}, Cint,
    ),
    pointer_from_objref(prob.cprob), ns
    )
    if ret != 0 
        error("IPOPT: Failed to construct problem.")
    end
    return prob
end
# function createProblem(n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
#     m::Int, g_L::Vector{Float64}, g_U::Vector{Float64},
#     nele_jac::Int, nele_hess::Int,
#     eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h = nothing)
#     @assert n == length(x_L) == length(x_U)
#     @assert m == length(g_L) == length(g_U)
#     # Wrap callbacks
#     eval_f_cb = @cfunction(eval_f_wrapper, Cint,
#     (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Cvoid}))
#     eval_g_cb = @cfunction(eval_g_wrapper, Cint,
#     (Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Cvoid}))
#     eval_grad_f_cb = @cfunction(eval_grad_f_wrapper, Cint,
#     (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Cvoid}))
#     eval_jac_g_cb = @cfunction(eval_jac_g_wrapper, Cint,
#     (Cint, Ptr{Float64}, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Float64}, Ptr{Cvoid}))
#     eval_h_cb = @cfunction(eval_h_wrapper, Cint,
#     (Cint, Ptr{Float64}, Cint, Float64, Cint, Ptr{Float64}, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Float64}, Ptr{Cvoid}))

#     ret = ccall((:CreateHiopProblem, libchiopInterface), Ptr{Cvoid},
#     (Cint, Ptr{Float64}, Ptr{Float64},  # Num vars, var lower and upper bounds
#     Cint, Ptr{Float64}, Ptr{Float64},  # Num constraints, con lower and upper bounds
#     Cint, Cint,                        # Num nnz in constraint Jacobian and in Hessian
#     Cint,                              # 0 for C, 1 for Fortran
#     Ptr{Cvoid}, Ptr{Cvoid},              # Callbacks for eval_f, eval_g
#     Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),  # Callbacks for eval_grad_f, eval_jac_g, eval_h
#     n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess, 1,
#     eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb, eval_h_cb)

#     if ret == C_NULL
#         error("IPOPT: Failed to construct problem.")
#     else
#         return HiopProblem(ret, n, m, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
#     end
# end

# TODO: Not even expose this? Seems dangerous, should just destruct
# the HiopProblem object via GC
function freeProblem(prob::HiopProblem)
    if prob.cprob.refcppHiop != C_NULL
        ccall(:hiop_destroyProblem, Cint, (Ptr{cHiopProblem},), pointer_from_objref(prob.cprob))
        prob.cprob.refcppHiop = C_NULL
    end
end

function solveProblem(prob::HiopProblem)
    final_objval = [0.0]
    @show prob.cprob.refcppHiop
    ret = ccall(:hiop_solveProblem, Cint, (Ptr{cHiopProblem}, ), pointer_from_objref(prob.cprob))
    # ret = ccall((:IpoptSolve, libipopt),
    # Cint, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Any),
    # prob.ref, prob.x, prob.g, final_objval, prob.mult_g, prob.mult_x_L, prob.mult_x_U, prob)
    # prob.obj_val = final_objval[1]
    # prob.status = Int(ret)

    return Int(ret)
end

include("MOI_wrapper.jl")

end # module
