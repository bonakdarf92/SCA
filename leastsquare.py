import cvxpy as cp
import numpy as np

m = 20
n = 15
np.random.seed(42)
A = np.random.randn(m,n)
b = np.random.randn(m)

x = cp.Variable(n)
cost = cp.sum_squares(A*x - b)
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

#print(prob.value)
#print(x.value)
#print(cp.norm(A*x - b, p = 2))

