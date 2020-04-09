include("../src/Hiop.jl")
using .Hiop
using LinearAlgebra

function eval_f(x, prob::HiopProblem) 
  # @show prob.ns
  obj = x[1]*(x[1] - 1.)
  for i in 2:prob.ns
    obj += x[i] * (x[i]-1.)
  end
  obj *= 0.5

  term2 = 0.0
  y = x[2*prob.ns+1:2*prob.ns + prob.nd]
  buf_y = prob.user_data.Q * y
  for i in 1:prob.nd
    term2 += buf_y[i] * y[i]
  end
  obj += 0.5 * term2
  # println("Term2: ", obj)

  s = x[prob.ns+1:prob.ns+prob.ns]
  term3 = s[1]*s[1]
  for i in 2:prob.ns
    term3 += s[i]*s[i]
  end
  obj += 0.5 * term3
  # println("Term3: ", obj)

  return obj
end

function eval_g(x, g)
  # Bad: g    = zeros(2)  # Allocates new array
  # OK:  g[:] = zeros(2)  # Modifies 'in place'
  g[1] = x[1]   * x[2]   * x[3]   * x[4]
  g[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
end

function eval_grad_f(x, grad_f)
  # Bad: grad_f    = zeros(4)  # Allocates new array
  # OK:  grad_f[:] = zeros(4)  # Modifies 'in place'
  grad_f[1] = x[1] * x[4] + x[4] * (x[1] + x[2] + x[3])
  grad_f[2] = x[1] * x[4]
  grad_f[3] = x[1] * x[4] + 1
  grad_f[4] = x[1] * (x[1] + x[2] + x[3])
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

n = 4
x_L = [1.0, 1.0, 1.0, 1.0]
x_U = [5.0, 5.0, 5.0, 5.0]

m = 2
g_L = [25.0, 40.0]
g_U = [2.0e19, 40.0]

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

nlp = createProblem(ns, n, x_L, x_U, m, g_L, g_U, 8, 10,
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
