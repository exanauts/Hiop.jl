include("../src/Hiop.jl")
using .Hiop
using LinearAlgebra

function eval_f(x::Vector{Float64}, prob::HiopProblem) 
  xrange = 1:prob.ns
  yrange = 2*prob.ns+1:3*prob.ns 
  srange = prob.ns+1:2*prob.ns
  return obj = 0.5 * ( sum(x[xrange] .* (x[xrange] .- 1.))
              + sum((prob.user_data.Q * x[yrange]) .* x[yrange])
              + sum(x[srange] .* x[srange])
              )
end

function eval_g(x::Vector{Float64}, idx_cons::Vector{Int64}, g::Vector{Float64}, prob::HiopProblem)
  xrange = 1:prob.ns
  yrange = 2*prob.ns+1:3*prob.ns 
  srange = prob.ns+1:2*prob.ns
  
  s = x[srange]
  isEq = false
  for row in 1:length(idx_cons)
    con_idx = idx_cons[row] + 1
    if con_idx <= prob.ns
      g[con_idx] = x[con_idx] + s[con_idx]
      isEq = true
    else
      conineq_idx = con_idx - prob.ns
      if conineq_idx == 1
        g[conineq_idx] = x[1]
        g[conineq_idx] += sum(x[srange])
        g[conineq_idx] += sum(x[yrange])
      end
      if conineq_idx == 2
        g[conineq_idx] = x[2]
        g[conineq_idx] += sum(x[yrange])
      end
      if conineq_idx == 3
        g[conineq_idx] = x[3]
        g[conineq_idx] += sum(x[yrange])
      end
    end
  end
  if isEq
    g[1:prob.ns] += prob.user_data.Md * x[yrange]
  end
  return Int32(1)
end

function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64}, prob::HiopProblem)
  xrange = 1:prob.ns
  yrange = 2*prob.ns+1:3*prob.ns 
  srange = prob.ns+1:2*prob.ns

  grad_f[xrange] .= x[xrange] .- 0.5
  grad_f[yrange] .= prob.user_data.Q * x[yrange]
  grad_f[srange] = x[srange]
  return Int32(1)
end

function eval_jac_g(mode::Vector{Symbol}, x::Vector{Float64}, idx_cons::Vector{Int64},
  iJacS::Vector{Int32}, jJacS::Vector{Int32}, MJacS::Vector{Float64}, 
  JacD::Vector{Float64}, prob::HiopProblem)
  if :Structure in mode
    nnzit = 1
    for i in 1:length(idx_cons)
      con_idx = idx_cons[i]
      if con_idx < prob.ns
        # x
        iJacS[nnzit] = con_idx
        jJacS[nnzit] = con_idx
        nnzit += 1
        # y
        iJacS[nnzit] = con_idx
        jJacS[nnzit] = con_idx + prob.ns
        nnzit += 1
      else
        if con_idx - prob.ns == 0
          # wrt x1
          iJacS[nnzit] = 0
          jJacS[nnzit] = 0
          nnzit += 1
          # wrt s
          for i in 1:prob.ns
            iJacS[nnzit] = 0
            jJacS[nnzit] = prob.ns + i - 1
            nnzit += 1
          end
        else
          # wrt x2 or x3
          @assert con_idx-prob.ns == 1 || con_idx-prob.ns == 2
          iJacS[nnzit] = con_idx - prob.ns
          jJacS[nnzit] = con_idx - prob.ns
          nnzit += 1
        end
      end
    end
    @assert nnzit == length(iJacS) + 1
  end
  # values for sparse Jacobian if requested by the solver
  if :Sparse in mode
    nnzit = 1
    for i in 1:length(idx_cons)
      con_idx = idx_cons[i]
      if con_idx < prob.ns
        # sparse Jacobian EQ w.r.t. x and s
        # x
        MJacS[nnzit] = 1.0
        nnzit += 1
        # s
        MJacS[nnzit] = 1.0
        nnzit += 1
      else
        if con_idx - prob.ns == 0
          # wrt x1
          MJacS[nnzit] = 1.0
          nnzit += 1
          for i in 1:prob.ns
            MJacS[nnzit] = 1.0
            nnzit += 1
          end
        else
          # wrt x2 or x3
          @assert con_idx-prob.ns == 1 || con_idx-prob.ns == 2
          MJacS[nnzit] = 1.0
          nnzit += 1
        end
      end
    end
    @assert nnzit == length(iJacS) + 1
  end
  if :Dense in mode
    # dense Jacobian w.r.t y
    isEq = false
    for i in 1:length(idx_cons)
      con_idx = idx_cons[i]
      if con_idx < prob.ns
        isEq = true
        @assert length(idx_cons) == prob.ns
        continue
      else
        # do an in place fill-in for the ineq Jacobian corresponding to e^T
        @assert con_idx - prob.ns == 0 || con_idx - prob.ns == 1 || con_idx - prob.ns == 2
        @assert length(idx_cons) == 3
        JacDmat = reshape(JacD, (3, prob.nd))
        for i in 1:prob.nd
          JacDmat[con_idx - prob.ns + 1, i] = 1
        end
      end
    end
    if isEq
      Mdvec = reshape(Md,(size(prob.user_data.Md,1) * size(prob.user_data.Md,2),))
      JacD .= Mdvec
    end
  end
  return Int32(1)
end

function eval_h(mode::Vector{Symbol}, x::Vector{Float64}, obj_factor::Float64, lambda::Vector{Float64}, 
                iHSS::Vector{Int32}, jHSS::Vector{Int32}, MHSS::Vector{Float64}, HDD::Vector{Float64}, 
                iHSD::Vector{Int32}, jHSD::Vector{Int32}, MHSD::Vector{Float64}, prob)
  @assert length(MHSS) == 2 * prob.ns
  @assert length(MHSD) == 0
  if :Structure in mode
    for i in 1:2*prob.ns
      iHSS[i] = i - 1
      jHSS[i] = i - 1
    end
  end
  if :Sparse in mode
    for i in 1:2*prob.ns 
      MHSS[i] = obj_factor
    end
  end
  if :Dense in mode
    Qvec = reshape(prob.user_data.Q, (prob.nd*prob.nd,))
    for i in 1:prob.nd^2
      HDD[i] = obj_factor * Qvec[i]
    end
  end
  return Int32(1)
end

struct User_data
  ns::Float64
  Q::Matrix{Float64}
  Md::Matrix{Float64}
  buf_y::Vector{Float64}
end


ns = 100

Q = Matrix{Float64}(undef, ns, ns)
Q .= 1e-8
for i in 1:ns  
  Q[i,i] += 2.
end
for i in 2:ns-1  
  Q[i,i+1] += 1.
  Q[i+1,i] += 1.
end

Md = Matrix{Float64}(undef, ns, ns)
Md .= -1.0
buf_y = Vector{Float64}(undef, ns)

user_data = User_data(ns, Q, Md, buf_y)

n = 3*ns
m = ns+3
nx_sparse = 2*ns;
nx_dense = ns;
nnz_sparse_Jaceq = 2*ns;
nnz_sparse_Jacineq = 1+ns+1+1;
nnz_sparse_Hess_Lagr_SS = 2*ns;
nnz_sparse_Hess_Lagr_SD = 0;

x_L = Vector{Float64}(undef, n)
x_U = Vector{Float64}(undef, n)

g_L = Vector{Float64}(undef, m)
g_U = Vector{Float64}(undef, m)

# x
# for(int i=0; i<ns; ++i) xlow_[i] = -1e+20;
for i in 1:ns
  x_L[i] = -1e+20
end
# s
# for(int i=ns; i<2*ns; ++i) xlow_[i] = 0.;
for i in ns+1:2*ns
  x_L[i] = 0.
end
# y 
# xupp_[2*ns] = -4.;
# for(int i=2*ns+1; i<n; ++i) xlow_[i] = -1e+20;
x_L[2*ns+1] = -4.
for i in 2*ns+2:3*ns
  x_L[i] = -1e+20
end

# x
# for(int i=0; i<ns; ++i) xupp_[i] = 3.;
for i in 1:ns
  x_U[i] = 3.
end
# s
# for(int i=ns; i<2*ns; ++i) xupp_[i] = +1e+20;
for i in ns+1:2*ns
  x_U[i] = 1e+20
end
# y
# xupp_[2*ns] = 4.;
# for(int i=2*ns+1; i<n; ++i) xupp_[i] = +1e+20;
x_U[2*ns+1] = 4.
for i in 2*ns+2:3*ns
  x_U[i] = 1e+20
end

g_L .= 0.0
g_U .= 0.0
g_L[end-2] = -2.;    g_U[end-2] = 2.
g_L[end-1] = -1e+20; g_U[end-1] = 2.
g_L[end]   = -2.;    g_U[end]   = 1e+20



nlp = HiopProblem(ns, ns,
                    Int32(nx_sparse), Int32(nx_dense), Int32(nnz_sparse_Jaceq), Int32(nnz_sparse_Jacineq), Int32(nnz_sparse_Hess_Lagr_SS), Int32(nnz_sparse_Hess_Lagr_SD),
                    n, x_L, x_U, 
                    m, g_L, g_U,
                    eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, user_data)
nlp.x0 .= 1.0

solveProblem(nlp)