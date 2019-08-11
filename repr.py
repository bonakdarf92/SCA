import numpy as np 
from pygsp import graphs, plotting, learning
import matplotlib.pyplot as plt
from matplotlib import gridspec
G = graphs.Sensor(N=256,distributed=True,seed=42)
G.compute_fourier_basis()
label_signal = np.copysign(np.ones(G.N),G.U[:,9])
#label_signal = np.random.randn(G.N)
#fig,axes=plt.subplot2grid(2,3,figsize=(12,4))
#gs = gridspec.GridSpec(2,)
ax1 = plt.subplot2grid((3,3),(0,0),colspan=1)
ax2 = plt.subplot2grid((3,3),(0,1),colspan=1)
ax3 = plt.subplot2grid((3,3),(0,2),colspan=1)
ax4 = plt.subplot2grid((3,3),(1,0),colspan=3)
ax5 = plt.subplot2grid((3,3),(2,0),colspan=3)
G.plot(label_signal,ax=ax1)
ax1.set_title('Original')

rs = np.random.RandomState(42)
M = rs.rand(G.N)
M = (M > 0.6).astype(float)

sigma = 0.1
sub_samp = M * (label_signal + sigma* rs.standard_normal(G.N))
G.plot(sub_samp,ax=ax2)
ax2.set_title('gemessene Signale')


import pyunlocbox
from SCA.stela import stela_lasso,soft_thresholding
gamma = 3
d = pyunlocbox.functions.dummy()
r = pyunlocbox.functions.norm_l1()
f = pyunlocbox.functions.norm_l2(w=M, y=sub_samp,lambda_=gamma)

G.compute_differential_operator()
L = G.D.T.toarray()
step = 0.999 / (1 + np.linalg.norm(L))
solver = pyunlocbox.solvers.mlfbf(L=L, step=step)
x0 = sub_samp.copy()
prob1 = pyunlocbox.solvers.solve([d, r, f], solver=solver,x0=x0,rtol=0,maxit=1000)
G.plot(prob1['sol'],ax=ax3)
ax3.set_title('Rekonstruiert Signale')
ax4.errorbar(range(0,len(label_signal)),label_signal,yerr=label_signal-prob1['sol'],ecolor='red')
#fig.tight_layout()
r = pyunlocbox.functions.norm_l2(A=L, tight=False)
step = 0.999 / np.linalg.norm(np.dot(L.T, L) + gamma * np.diag(M), 2)
solver = pyunlocbox.solvers.gradient_descent(step=step)
x0 = sub_samp.copy()
prob2 = pyunlocbox.solvers.solve([r, f], solver=solver,x0=x0, rtol=0, maxit=1000)
ax5.errorbar(range(0,len(label_signal)),label_signal,yerr=label_signal-prob2['sol'],ecolor='red')
mu = 10000# * np.linalg.norm(np.dot(x0,G.L),np.inf)
obj,x_s,er = stela_lasso(G.L.T.toarray(),x0,mu)

plt.show()
