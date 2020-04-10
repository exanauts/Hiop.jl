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

function eval_grad_f(x::Vector{Float64}, grad_f, prob::HiopProblem)
  xrange = 1:prob.ns
  yrange = 2*prob.ns+1:3*prob.ns 
  srange = prob.ns+1:2*prob.ns

  grad_f[xrange] .= x[xrange] .- 0.5
  grad_f[yrange] .= prob.user_data.Q * x[yrange]
  grad_f[srange] = x[srange]
  return Int32(1)
end

function eval_jac_g(x, mode, rows, cols, values)
  if mode == :Structure
    # Constraint (row) 1
    rows[1] = 1; cols[1] = 1
    rows[2] = 1; cols[2] = 2
    rows[3] = 1; cols[3] = 3
    rows[4] = 1; cols[4] = 4
    # Constraint (row) 2
    rows[5] = 2; cols[5] = 1
    rows[6] = 2; cols[6] = 2
    rows[7] = 2; cols[7] = 3
    rows[8] = 2; cols[8] = 4
  else
    # Constraint (row) 1
    values[1] = x[2]*x[3]*x[4]  # 1,1
    values[2] = x[1]*x[3]*x[4]  # 1,2
    values[3] = x[1]*x[2]*x[4]  # 1,3
    values[4] = x[1]*x[2]*x[3]  # 1,4
    # Constraint (row) 2
    values[5] = 2*x[1]  # 2,1
    values[6] = 2*x[2]  # 2,2
    values[7] = 2*x[3]  # 2,3
    values[8] = 2*x[4]  # 2,4
  end
end

function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
  if mode == :Structure
    # Symmetric matrix, fill the lower left triangle only
    idx = 1
    for row = 1:4
      for col = 1:row
        rows[idx] = row
        cols[idx] = col
        idx += 1
      end
    end
  else
    # Again, only lower left triangle
    # Objective
    values[1] = obj_factor * (2*x[4])  # 1,1
    values[2] = obj_factor * (  x[4])  # 2,1
    values[3] = 0                      # 2,2
    values[4] = obj_factor * (  x[4])  # 3,1
    values[5] = 0                      # 3,2
    values[6] = 0                      # 3,3
    values[7] = obj_factor * (2*x[1] + x[2] + x[3])  # 4,1
    values[8] = obj_factor * (  x[1])  # 4,2
    values[9] = obj_factor * (  x[1])  # 4,3
    values[10] = 0                     # 4,4

    # First constraint
    values[2] += lambda[1] * (x[3] * x[4])  # 2,1
    values[4] += lambda[1] * (x[2] * x[4])  # 3,1
    values[5] += lambda[1] * (x[1] * x[4])  # 3,2
    values[7] += lambda[1] * (x[2] * x[3])  # 4,1
    values[8] += lambda[1] * (x[1] * x[3])  # 4,2
    values[9] += lambda[1] * (x[1] * x[2])  # 4,3

    # Second constraint
    values[1]  += lambda[2] * 2  # 1,1
    values[3]  += lambda[2] * 2  # 2,2
    values[6]  += lambda[2] * 2  # 3,3
    values[10] += lambda[2] * 2  # 4,4
  end
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



nlp = createProblem(ns,
                    Int32(nx_sparse), Int32(nx_dense), Int32(nnz_sparse_Jaceq), Int32(nnz_sparse_Jacineq), Int32(nnz_sparse_Hess_Lagr_SS), Int32(nnz_sparse_Hess_Lagr_SD),
                    n, x_L, x_U, 
                    m, g_L, g_U, 8, 10,
                    eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, user_data)

# prob.x = [1.0, 5.0, 5.0, 1.0]
# status = solveProblem(prob)

# println(Ipopt.ApplicationReturnStatus[status])
# println(prob.x)
# println(prob.obj_val)

# nlp = Ptr{Nothing}()
# @show nlp
solveProblem(nlp);
# destroyProblem(nlp);
