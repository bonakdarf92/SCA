import numpy as np 
from pygsp import graphs, plotting 
import matplotlib.pyplot as plt
 
G = graphs.Sensor(N=256,distribute=True,seed=42)
G.compute_fourier_basis()
label_signal = np.copysign(np.ones(G.N),G.U[:,3])
test = G.plot_signal(label_signal)
plt.show()

rs = np.random.RandomState(42)

M = rs.rand(G.N)
M = (M>0.6).astype(float)
sigma = 0.1
subsampled_noisy_label_signal = M * (label_signal + sigma * rs.standard_normal(G.N))

test2 = G.plot_signal(subsampled_noisy_label_signal)
plt.show()

import pyunlocbox as pyb 
gamma = 3.0 
d = pyb.functions.dummy()
r = pyb.functions.norm_l1()
f = pyb.functions.norm_l2(w=M,y=subsampled_noisy_label_signal,lambda_=gamma)
G.compute_differential_operator()
L = G.D.toarray()
step = 0.999 / (1+np.linalg.norm(L))
solver = pyb.solvers.mlfbf(L=L,step=step)
x0 = subsampled_noisy_label_signal.copy()
prob1 = pyb.solvers.solve([d,r,f],solver=solver,x0=x0,rtol=0,maxit=1000)
test3 = G.plot_signal(prob1['sol'])
plt.show()

r = pyb.functions.norm_l2(A=L,tight=False)

step = 0.999/np.linalg.norm(np.dot(L.T,L)+gamma*np.diag(M),2)
solver = pyb.solvers.gradient_descent(step=step)
x0 = subsampled_noisy_label_signal.copy()
prob2 = pyb.solvers.solve([r,f],solver=solver,x0=x0,rtol=0,maxit=1000)
test4 = G.plot_signal(prob2['sol'])
plt.show()
