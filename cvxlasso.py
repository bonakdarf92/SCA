import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
#import gurobipy


# Problem data
n = 200
m = 400
np.random.seed(20)

A = np.random.randn(n,m)
b = np.random.randn(n)
v = np.random.normal(0,0.05,n)
gamma = cp.Parameter(nonneg=True)

# Construct problem
x = cp.Variable(m)
measure = A*x + v 
error = cp.sum_squares(A*x - measure)
obj = cp.Minimize(error + gamma * cp.norm(x,1))
prob = cp.Problem(obj)

# Construct trade off curve 

sq_penalty = []
l1_penalty = []
x_values = []
gamma_vals = np.logspace(-4, 6)
for val in gamma_vals:
    gamma.value = val
    prob.solve()
    # Use expr.value to get the numerical value of of the problem
    sq_penalty.append(error.value)
    l1_penalty.append(cp.norm(x,1).value)
    x_values.append(x.value)

plt.rcParams.update({"pgf.texsystem":"pdflatex",
        "pgf.preamble":[
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{cmbright}",]
        })
print(cp.installed_solvers())

plt.figure(figsize=(6,10))

#plt.rc('text',usetex=True)
plt.rc("font", family="serif")
#plt.figure(figsize=(6,10))
plt.subplot(211)
plt.plot(l1_penalty, sq_penalty)
plt.xlabel(r"$\|x\|_1$", fontsize=16)
plt.ylabel(r"$\|Ax-b\|^2$",fontsize=16)
plt.title('Tradeoff Curve for Lasso', fontsize=16)
#plt.show()

# Plot entries of x vs gamma
plt.subplot(212)
for i in range(m):
    plt.plot(gamma_vals, [xi[i] for xi in x_values])
plt.xlabel(r"$\gamma$", fontsize=16)
plt.ylabel(r"$x_{i}$",fontsize=16)
plt.xscale('log')
plt.title(r"Entries of x vs. $\gamma$", fontsize=16)
plt.tight_layout()
plt.show()

