import cvxpy as cp
import numpy as np

n = 3
p = 3
np.random.seed(1)
C = np.random.randn(n,n)
A = []
b = []
for i in range(p):
    A.append(np.random.randn(n,n))
    b.append(np.random.randn())

X = cp.Variable((n,n), symmetric=True)
constraints = [X >> 0]
constraints += [cp.trace(A[i]@X) == b[i] for i in range(p)]
prob = cp.Problem(cp.Minimize(cp.trace(C@X)),constraints)
prob.solve()

print("optimum", prob.value)
print(X.value)
