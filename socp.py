import cvxpy as cp
import numpy as np

m = 3
n = 10
p = 5
n_i = 5
np.random.seed(2)
f = np.random.randn(n)
A = []
b = []
c = []
d = []
x0 = np.random.randn(n)
for i in range(m):
    A.append(np.random.randn(n_i, n))
    b.append(np.random.randn(n_i))
    c.append(np.random.randn(n))
    d.append(np.linalg.norm(A[i]@x0 + b, 2) - c[i].T@x0)
F = np.random.randn(p,n)
g = F@x0

# Define and solve the cvxpy problem
x = cp.Variable(n)

# We use cp.SOC(t,x) to create the SOC constraint 
soc_constraint = [cp.SOC(c[i].T@x + d[i], A[i]@x + b[i]) for i in range(m)]
prob = cp.Problem(cp.Minimize(f.T@x), soc_constraint + [F@x == g])
prob.solve()
print("The optimal value is", prob.value)
print("A solution x is")
print(x.value)
for i in range(m):
    print("SOC constraint %i dual variable solution" % i)
    print(soc_constraint[i].dual_value)

