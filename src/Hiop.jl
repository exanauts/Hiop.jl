module Hiop
using Libdl
using LinearAlgebra

export solveProblem
export HiopProblem

function __init__()
  try
    path_to_lib = ENV["JULIA_HIOP_LIBRARY_PATH"]
    Libdl.dlopen("libf77blas.so", Libdl.RTLD_GLOBAL)
    Libdl.dlopen("liblapack.so", Libdl.RTLD_GLOBAL)
    Libdl.dlopen("libhiop.so", Libdl.RTLD_GLOBAL)
    Libdl.dlopen(joinpath(dirname(@__FILE__), "../deps/libchiopInterface.so"), Libdl.RTLD_GLOBAL)
  catch
    @warn("Could not load HiOp shared library. Make sure the ENV variable 'JULIA_HIOP_LIBRARY_PATH' points to its location.")
    rethrow()
  end
end

mutable struct cHiopProblem
  refcppHiop::Ptr{Cvoid}
  jprob::Ptr{Cvoid}
  get_starting_point::Ptr{Cvoid}
  get_prob_sizes::Ptr{Cvoid}
  get_vars_info::Ptr{Cvoid}
  get_cons_info::Ptr{Cvoid}
  eval_f::Ptr{Cvoid}
  eval_grad_f::Ptr{Cvoid}
  eval_cons::Ptr{Cvoid}
  get_sparse_dense_blocks_info::Ptr{Cvoid}
  eval_Jac_cons::Ptr{Cvoid}
  eval_Hess_Lagr::Ptr{Cvoid}
end
  # Forward declarations coming after the constructor
  function get_starting_point_wrapper end
  function get_sparse_dense_blocks_info_wrapper end
  function get_prob_sizes_wrapper end
  function get_vars_info_wrapper end
  function get_cons_info_wrapper end
  function eval_f_wrapper end
  function eval_grad_f_wrapper end
  function eval_g_wrapper end
  function eval_jac_g_wrapper end
  function eval_h_wrapper end
  function freeProblem end

mutable struct HiopProblem
  cprob::cHiopProblem  # Reference to the C data structure
  n::Int64  # Num vars
  m::Int64  # Num cons
  ns::Int64 # Sparse whatever
  nd::Int64 # Dense whatever
  nx_sparse::Int32
  nx_dense::Int32 
  nnz_sparse_Jaceq::Int32
  nnz_sparse_Jacineq::Int32
  nnz_sparse_Hess_Lagr_SS::Int32
  nnz_sparse_Hess_Lagr_SD::Int32
  x::Vector{Float64}  # Starting and final solution
  x_L::Vector{Float64}  # Starting and final solution
  x_U::Vector{Float64}  # Starting and final solution
  g::Vector{Float64}  # Final constraint values
  g_L::Vector{Float64}  # Final constraint values
  g_U::Vector{Float64}  # Final constraint values
  mult_g::Vector{Float64} # lagrange multipliers on constraints
  mult_x_L::Vector{Float64} # lagrange multipliers on lower bounds
  mult_x_U::Vector{Float64} # lagrange multipliers on upper bounds
  obj_val::Float64  # Final objective
  status::Int  # Final status

  # Callbacks
  eval_f::Function
  eval_g::Function
  eval_grad_f::Function
  eval_jac_g::Function
  eval_h  # Can be nothing
  x0::Vector{Float64}
  user_data::Any 
  

  function HiopProblem(ns::Int, nd::Int, nx_sparse::Int32, nx_dense::Int32,
    nnz_sparse_Jaceq::Int32, nnz_sparse_Jacineq::Int32, nnz_sparse_Hess_Lagr_SS::Int32, nnz_sparse_Hess_Lagr_SD::Int32,
    n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
    m::Int, g_L::Vector{Float64}, g_U::Vector{Float64},
    eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h = nothing, user_data = nothing)
    # Wrap callbacks
    prob = new(cHiopProblem(C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL), 
                n, m, ns, nd, 
                nx_sparse, nx_dense, nnz_sparse_Jaceq, nnz_sparse_Jacineq, nnz_sparse_Hess_Lagr_SS, nnz_sparse_Hess_Lagr_SD,
                zeros(Float64, n), x_L, x_U, 
                zeros(Float64, m), g_L, g_U,
                zeros(Float64,m),
                zeros(Float64,n), zeros(Float64,n), 0.0, 0,
                eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, zeros(Float64, n), user_data)

    prob.cprob.get_starting_point = @cfunction(get_starting_point_wrapper, Cint,
                    (Clonglong, Ptr{Cdouble}, Ptr{Cvoid}))
    prob.cprob.get_sparse_dense_blocks_info = @cfunction(get_sparse_dense_blocks_info_wrapper, Cint,
                    (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}))
    prob.cprob.get_prob_sizes = @cfunction(get_prob_sizes_wrapper, Cint,
                    (Ptr{Clonglong}, Ptr{Clonglong}, Ptr{Cvoid}))
    prob.cprob.get_vars_info = @cfunction(get_vars_info_wrapper, Cint,
                    (Ptr{Clonglong}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
    prob.cprob.get_cons_info = @cfunction(get_cons_info_wrapper, Cint,
                    (Ptr{Clonglong}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
    prob.cprob.eval_f = @cfunction(eval_f_wrapper, Cint,
                    (Clonglong, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cvoid}))
    prob.cprob.eval_grad_f = @cfunction(eval_grad_f_wrapper, Cint,
                    (Clonglong, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cvoid}))
    prob.cprob.eval_cons = @cfunction(eval_g_wrapper, Cint,
                    (Clonglong, Clonglong, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cvoid}))
    prob.cprob.eval_Jac_cons = @cfunction(eval_jac_g_wrapper, Cint,
                    (Clonglong, Clonglong, Ptr{Cdouble}, Cint, Clonglong, Clonglong,
                    Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
    prob.cprob.eval_Hess_Lagr = @cfunction(eval_h_wrapper, Cint,
                    (Clonglong, Clonglong, 
                    Ptr{Cdouble}, Cint, Cdouble,
                    Ptr{Cdouble}, Cint, 
                    Clonglong, Clonglong,
                    Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, 
                    Ptr{Cdouble}, 
                    Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cvoid}))
    prob.cprob.jprob = pointer_from_objref(prob)
    @show prob.cprob.jprob
    ret = ccall(:hiop_createProblem, Cint, (Ptr{cHiopProblem}, Cint,), pointer_from_objref(prob.cprob), ns)
    if ret != 0 
        error("HiOp: Failed to construct problem.")
    end
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

function get_starting_point_wrapper(n::Clonglong, x0_::Ptr{Cdouble}, prob_::Ptr{Cvoid})
  prob = unsafe_pointer_to_objref(prob_)::HiopProblem
  x0 = unsafe_wrap(Array{Float64}, x0_, prob.n)
  x0 .= prob.x0
  return Int32(1)
end
function get_prob_sizes_wrapper(n_::Ptr{Clonglong}, m_::Ptr{Clonglong}, prob_::Ptr{Cvoid})
  prob = unsafe_pointer_to_objref(prob_)::HiopProblem
  n = unsafe_wrap(Array{Int64}, n_, 1)
  m = unsafe_wrap(Array{Int64}, m_, 1)
  n[1] = prob.n
  m[1] = prob.m
  return Int32(1)
end

function get_vars_info_wrapper(n_::Ptr{Clonglong}, xlow_::Ptr{Cdouble}, xupp_::Ptr{Cdouble}, prob_::Ptr{Cvoid})
  prob = unsafe_pointer_to_objref(prob_)::HiopProblem
  xlow = unsafe_wrap(Array{Float64}, xlow_, prob.n)
  xupp = unsafe_wrap(Array{Float64}, xupp_, prob.n)
  xlow .= prob.x_L
  xupp .= prob.x_U
  return Int32(1)
end

function get_cons_info_wrapper(m_::Ptr{Clonglong}, clow_::Ptr{Cdouble}, cupp_::Ptr{Cdouble}, prob_::Ptr{Cvoid})
  prob = unsafe_pointer_to_objref(prob_)::HiopProblem
  clow = unsafe_wrap(Array{Float64}, clow_, prob.m)
  cupp = unsafe_wrap(Array{Float64}, cupp_, prob.m)
  clow .= prob.g_L
  cupp .= prob.g_U
  return Int32(1)
end

function get_sparse_dense_blocks_info_wrapper(nx_sparse_::Ptr{Cint}, nx_dense_::Ptr{Cint},
    nnz_sparse_Jaceq_::Ptr{Cint}, nnz_sparse_Jacineq_::Ptr{Cint},
    nnz_sparse_Hess_Lagr_SS_::Ptr{Cint}, 
    nnz_sparse_Hess_Lagr_SD_::Ptr{Cint}, 
    prob_::Ptr{Cvoid})
  prob = unsafe_pointer_to_objref(prob_)::HiopProblem
  nx_sparse = unsafe_wrap(Array{Int32}, nx_sparse_, 1)
  nx_dense = unsafe_wrap(Array{Int32}, nx_dense_, 1)
  nnz_sparse_Jaceq = unsafe_wrap(Array{Int32}, nnz_sparse_Jaceq_, 1)
  nnz_sparse_Jacineq = unsafe_wrap(Array{Int32}, nnz_sparse_Jacineq_, 1)
  nnz_sparse_Hess_Lagr_SS = unsafe_wrap(Array{Int32}, nnz_sparse_Hess_Lagr_SS_, 1)
  nnz_sparse_Hess_Lagr_SD = unsafe_wrap(Array{Int32}, nnz_sparse_Hess_Lagr_SD_, 1)

  nx_sparse[1] = prob.nx_sparse
  nx_dense[1] = prob.nx_dense
  nnz_sparse_Jaceq[1] = prob.nnz_sparse_Jaceq
  nnz_sparse_Jacineq[1] = prob.nnz_sparse_Jacineq
  nnz_sparse_Hess_Lagr_SS[1] = prob.nnz_sparse_Hess_Lagr_SS
  nnz_sparse_Hess_Lagr_SD[1] = prob.nnz_sparse_Hess_Lagr_SD
  return Int32(1)
end

function eval_f_wrapper(n::Clonglong, x_ptr::Ptr{Float64}, new_x::Cint, obj_ptr::Ptr{Float64}, prob_::Ptr{Cvoid})
  # Extract Julia the problem from the pointer
  prob = unsafe_pointer_to_objref(prob_)::HiopProblem
  # Calculate the new objective
  new_obj = convert(Float64, prob.eval_f(unsafe_wrap(Array,x_ptr, Int(n)), prob))::Float64
  # Fill out the pointer
  unsafe_store!(obj_ptr, new_obj)
  # Done
  return Int32(1)
end

# Constraints (eval_g)
function eval_g_wrapper(n::Clonglong, m::Clonglong, x_ptr::Ptr{Float64}, new_x::Cint, g_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})
  # Extract Julia the problem from the pointer
  prob = unsafe_pointer_to_objref(user_data)::HiopProblem
  x = unsafe_wrap(Array{Float64}, x_ptr, prob.n)
  # Calculate the new constraint values
  new_g = unsafe_wrap(Array,g_ptr, Int(m))
  prob.eval_g(x, new_g, prob)
  # Done
  return Int32(1)
end

# Objective gradient (eval_grad_f)
function eval_grad_f_wrapper(n::Clonglong, x_ptr::Ptr{Float64}, new_x::Cint, grad_f_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})
  # Extract Julia the problem from the pointer
  prob = unsafe_pointer_to_objref(user_data)::HiopProblem
  # Calculate the gradient
  new_grad_f = unsafe_wrap(Array,grad_f_ptr, Int(n))
  prob.eval_grad_f(unsafe_wrap(Array,x_ptr, Int(n)), new_grad_f, prob)
  # Done
  return Int32(1)
end

# Jacobian (eval_jac_g)
function eval_jac_g_wrapper(n::Clonglong, m::Clonglong,
    x_ptr::Ptr{Cdouble}, new_x::Cint, nsparse::Clonglong, ndense::Clonglong, nnzJacS::Cint, iJacS_ptr::Ptr{Cint}, 
    jJacS_ptr::Ptr{Cint}, MJacS_ptr::Ptr{Cdouble}, JacD_ptr::Ptr{Cdouble}, user_data::Ptr{Cvoid})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::HiopProblem
    # Determine mode
    mode = Vector{Symbol}()
    if iJacS_ptr != C_NULL && jJacS_ptr != C_NULL
      push!(mode, :Structure)
    end
    if MJacS_ptr != C_NULL
        push!(mode, :Sparse)
    end
    if JacD_ptr != C_NULL
        push!(mode, :Dense)
    end
    x = unsafe_wrap(Array, x_ptr, Int(n))
    iJacS = unsafe_wrap(Array, iJacS_ptr, Int(nnzJacS))
    jJacS = unsafe_wrap(Array, jJacS_ptr, Int(nnzJacS))
    MJacS = unsafe_wrap(Array, MJacS_ptr, Int(nnzJacS))
    # @show n, m, nsparse, ndense, prob.ns, prob.nd
    JacD = unsafe_wrap(Array, JacD_ptr, prob.m * prob.nd)
    prob.eval_jac_g(mode, x, iJacS, jJacS, MJacS, JacD, prob)
    # Done
    return Int32(1)
end

# Hessian
function eval_h_wrapper(n::Clonglong, m::Clonglong,
  x_ptr::Ptr{Cdouble}, new_x::Cint, obj_factor::Cdouble,
  lambda_ptr::Ptr{Cdouble}, new_lambda::Cint, 
  nsparse::Clonglong, ndense::Clonglong, 
  nnzHSS::Cint, iHSS_ptr::Ptr{Cint}, jHSS_ptr::Ptr{Cint}, MHSS_ptr::Ptr{Cdouble}, 
  HDD_ptr::Ptr{Cdouble}, 
  nnzHSD::Cint, iHSD_ptr::Ptr{Cint}, jHSD_ptr::Ptr{Cint}, MHSD_ptr::Ptr{Cdouble},
  user_data::Ptr{Cvoid})
  # Extract Julia the problem from the pointer
  prob = unsafe_pointer_to_objref(user_data)::HiopProblem
  # Determine mode
  mode = Vector{Symbol}()
  if iHSS_ptr != C_NULL && jHSS_ptr != C_NULL
      push!(mode, :Structure)
  end
  if MHSS_ptr != C_NULL
      push!(mode, :Sparse)
  end
  if HDD_ptr != C_NULL
      push!(mode, :Dense)
  end
  x = unsafe_wrap(Array, x_ptr, Int(n))
  lambda = unsafe_wrap(Array, lambda_ptr, Int(m))
  iHSS = unsafe_wrap(Array, iHSS_ptr, Int(nnzHSS))
  jHSS = unsafe_wrap(Array, jHSS_ptr, Int(nnzHSS))
  MHSS = unsafe_wrap(Array, MHSS_ptr, Int(nnzHSS))
  HDD = unsafe_wrap(Array, HDD_ptr, Int(prob.nd * prob.nd))
  iHSD = unsafe_wrap(Array, iHSD_ptr, Int(nnzHSD))
  jHSD = unsafe_wrap(Array, jHSD_ptr, Int(nnzHSD))
  MHSD = unsafe_wrap(Array, MHSD_ptr, Int(nnzHSD))
  prob.eval_h(mode, x, obj_factor, lambda, iHSS, jHSS, MHSS, HDD, iHSD, jHSD, MHSD, prob)
  # Done
  return Int32(1)
end

###########################################################################
# C function wrappers
###########################################################################

function freeProblem(prob::HiopProblem)
    if prob.cprob.refcppHiop != C_NULL
        ccall(:hiop_destroyProblem, Cint, (Ptr{cHiopProblem},), pointer_from_objref(prob.cprob))
        prob.cprob.refcppHiop = C_NULL
    end
end

function solveProblem(prob::HiopProblem)
    final_objval = [0.0]
    ret = ccall(:hiop_solveProblem, Cint, (Ptr{cHiopProblem}, ), pointer_from_objref(prob.cprob))
    return Int(ret)
end

include("MOI_wrapper.jl")

end # module
