import numpy as np
import matplotlib.pyplot as plt
#from stela import stela_lasso, soft_thresholding
"""
N = 1203; # number of rows of A (measurements)
K = 1203; # number of columns of A (features)


'''disable the following two lines to input your own number of measurements and features'''
#N = int(input("Please enter the number of measurements: "))
#K = int(input("Please enter the number of features: "))

'''generate measurements: y = A * x0 + v'''
np.random.seed(0)
A = np.random.normal(0, 0.1, (N, K))
A = np.vstack((A,A,A,A,A))
## =============================================================================
## normalize each row of A
#A_row_norm = np.linalg.norm(A, axis=1)
#A_row_norm_matrix = np.matrix.transpose(np.kron(A_row_norm,np.ones((K,1))))
#A = np.divide(A, A_row_norm_matrix)
## =============================================================================

'''generate the sparse vector'''
density          = 1   # density of the sparse signal
x0               = np.zeros(K)
x0_positions     = np.random.choice(np.arange(K), int(K * density), replace = False)
x0[x0_positions] = np.random.normal(0, 1, int(K * density))

'''generate the noise'''
sigma = 0.05 # noise standard deviation
v     = np.random.normal(1.5, sigma, 5*N) # noise

'''generate the noisy measurement'''
y  = np.dot(A, x0) + v # measurement
y[np.random.choice(np.arange(K),int(K*0.6),replace=False)] = 0

'''regularization gain'''
mu = 0.005*np.linalg.norm(np.dot(y, A), np.inf)

'''call STELA'''
MaxIter = 20 # maximum number of iterations, optional input
objval, x, error = stela_lasso(A, y, mu, MaxIter)

'''plot output'''
'''compare the original signal and the estimated signal'''
plt.plot(np.linspace(1, K, K), x0, 'b-x', label = "original signal")
plt.plot(np.linspace(1, K, K), x, 'r-.', label = "estimated signal")
plt.legend(bbox_to_anchor = (1.05, 1), loc = 1, borderaxespad = 0.)
plt.xlabel("index")
plt.ylabel("coefficient")
plt.show()

'''number of iterations vs. objective function value'''
plt.plot(np.linspace(0, objval.size-1, objval.size), objval, 'r-')
plt.xlabel("number of iterations")
plt.ylabel("objective function value")
plt.show()

'''number of iterations vs. solution precision'''
plt.plot(np.linspace(0, error.size-1, error.size), error, 'r-')
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.yscale('log')
plt.show()
"""
"""
import numpy as np
import matplotlib.pyplot as plt
from stela import stela_lasso, soft_thresholding

N = 1000; # number of rows of A (measurements)
K = 2; # number of columns of A (features)


'''disable the following two lines to input your own number of measurements and features'''
#N = int(input("Please enter the number of measurements: "))
#K = int(input("Please enter the number of features: "))

'''generate measurements: y = A * x0 + v'''
np.random.seed(0)
A = np.random.normal(0, 0.1, (N, K))
## =============================================================================
## normalize each row of A
#A_row_norm = np.linalg.norm(A, axis=1)
#A_row_norm_matrix = np.matrix.transpose(np.kron(A_row_norm,np.ones((K,1))))
#A = np.divide(A, A_row_norm_matrix)
## =============================================================================

'''generate the sparse vector'''
density          = 1   # density of the sparse signal
x0               = np.zeros(K)
x0_positions     = np.random.choice(np.arange(K), int(K * density), replace = False)
x0[x0_positions] = np.random.normal(0, 1, int(K * density))

'''generate the noise'''
sigma = 0.05; # noise standard deviation
v     = np.random.normal(0, sigma, N) # noise

'''generate the noisy measurement'''
y  = np.dot(A, x0) + v # measurement

'''regularization gain'''
mu = 0.001*np.linalg.norm(np.dot(y, A), np.inf)

'''call STELA'''
MaxIter = 5000 # maximum number of iterations, optional input
objval, x, error = stela_lasso(A, y, mu, MaxIter)

'''plot output'''
'''compare the original signal and the estimated signal'''
plt.plot(np.linspace(1, K, K), x0, 'b-x', label = "original signal")
plt.plot(np.linspace(1, K, K), x, 'r-.', label = "estimated signal")
plt.legend(bbox_to_anchor = (1.05, 1), loc = 1, borderaxespad = 0.)
plt.xlabel("index")
plt.ylabel("coefficient")
plt.show()

'''number of iterations vs. objective function value'''
plt.plot(np.linspace(0, objval.size-1, objval.size), objval, 'r-')
plt.xlabel("number of iterations")
plt.ylabel("objective function value")
plt.show()

'''number of iterations vs. solution precision'''
plt.plot(np.linspace(0, error.size-1, error.size), error, 'r-')
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.yscale('log')
plt.show()
"""

from stela import *
import numpy as np
from DarmstadtNetwork import DarmstadtNetwork
import pygsp as ps 
darmi = DarmstadtNetwork()
A = darmi.remove_diagsLoops(direction="directed").toarray()
D_g = ps.graphs.Graph(A) 
xs,ys,_ = darmi.get_ids()
D_g.set_coordinates([[v,z] for v,z in zip(xs,ys)])
D_g.compute_differential_operator()
D_g.compute_fourier_basis()
plt.show()
D1 = D_g.D.toarray() 
L = D_g.L.toarray()
#A = np.random.normal(0,0.1,(1000,100))
#N = 1000
#K = 100
density = 0.7
mean = 6
#A = np.concatenate((L,L),axis=0)  # doppelt Eintrag
A = D1.T
N,K = A.shape
x0 = np.zeros(K)
x0_positions = np.random.choice(np.arange(K),int(K*density),replace=False)

x0[x0_positions] = np.random.normal(mean,1,int(K*density))  # sparse signal
#x0 = np.random.normal(0,1,int(K))
#x_up = np.concatenate((x0,x0),axis=0)   # doppelt Eintrag
fig,ax = plt.subplots(1,2,figsize=(12,8))
plt.set_cmap('seismic_r')
D_g.plot(x0,vertex_size=30,ax=ax[0])
ax[0].set_title('Original')
#plt.set_cmap('coolwarm')
#plt.show()
#D_g.plot()
sigma = 0.001
v = np.random.normal(0,sigma,N)
#A = np.concatenate((A,A),axis=1)       # doppelt Eintrag
#v = np.concatenate((v,v),axis=0)
rs = np.random.RandomState(42)
M = (rs.rand(D_g.N,D_g.N) > 0.4).astype(float)
y = np.dot(A,x0) + v
mask = np.ones(len(x0),np.bool)
mask[x0_positions] = True
selec = np.random.choice(x0[mask],int(K*density*0.7),replace=False).astype(int)
y[selec] = 0
#y = np.dot(M,x0) + v

mu = 0.00008 * np.linalg.norm(np.dot(y,A),np.inf)
theta = 50
maxiter = 25000
objval, x, error = stela_cappedL1(A,y,mu,theta,maxiter)
offset = np.absolute(np.mean(x[x<np.amin(x)+1]))

D_g.plot(x+offset, vertex_size=30,ax=ax[1])
ax[1].set_title('Reconstruct')
ax[1].set_xlim([8.613,8.69])
ax[1].set_ylim([49.846,49.882])
ax[0].set_xlim([8.613,8.69])
ax[0].set_ylim([49.846,49.882])
fig,ax = plt.subplots(1,1,figsize=(12,6))
plt.set_cmap('seismic_r')
D_g.plot(x0-(x+offset),vertex_size=30,ax=ax)
ax.set_title('Differenz')
ax.set_xlim([8.613,8.69])
ax.set_ylim([49.846,49.882])

fig.tight_layout()
plt.show()

'''plot output'''
'''compare the original signal and the estimated signal'''
plt.plot(np.linspace(1, K, K), x0, 'bx', label = "original signal")
plt.plot(np.linspace(1, K, K), x+offset, 'ro',mfc='none', label = "estimated signal")
plt.legend(bbox_to_anchor = (1.05, 1), loc = 1, borderaxespad = 0.)
plt.xlabel("index")
plt.ylabel("coefficient")
plt.title('Sparsity x: {}, missing y {}, offsest {}'.format(len(x0_positions),len(selec),offset))
plt.show()

'''number of iterations vs. objective function value'''
plt.plot(np.linspace(0, objval.size-1, objval.size), objval, 'r-')
plt.xlabel("number of iterations")
plt.ylabel("objective function value")
plt.show()

'''number of iterations vs. solution precision'''
plt.plot(np.linspace(0, error.size-1, error.size), error, 'r-')
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.yscale('log')
plt.show()
import numpy as np
from matplotlib import pyplot as plt
import pygsp as pg
from DarmstadtNetwork import DarmstadtNetwork
dtown = DarmstadtNetwork()
xs, ys, _ = dtown.get_ids()
A1 = dtown.sparse_adj 
A2 = dtown.remove_diagsLoops(direction="directed")  
G = pg.graphs.Graph(A2)
G.set_coordinates([[v,z] for v,z in zip(xs,ys)]) 
G.compute_laplacian('combinatorial')
G.compute_fourier_basis()
G.compute_differential_operator() 
fig, axes = plt.subplots(1,5,figsize=(12,4))
def plot_eigenvectors(G,axes):
    limits = [f(G.U) for f in (np.min, np.max)]
    for i, ax in enumerate(axes):
        G.plot(G.U[:,i], limits=limits, colorbar=False, vertex_size=30,ax=ax)
        energy = abs(G.dirichlet_energy(G.U[:,i]))
        ax.set_title(r'$u^T L u = {}$'.format(energy))
        ax.set_axis_off()

plot_eigenvectors(G, axes)
plt.show()

scales = [10, 3, 0]
limit = 1

fig, axes = plt.subplots(2, len(scales), figsize=(12, 4))
fig.subplots_adjust(hspace=0.5)


x0 = np.random.RandomState(1).normal(size=G.N)
for i, scale in enumerate(scales):
    g = pg.filters.Heat(G, scale)
    x = g.filter(x0).squeeze()
    x /= np.linalg.norm(x)
    x_hat = G.gft(x).squeeze()

    assert np.all((-limit < x) & (x < limit))
    G.plot(x, limits=[-limit, limit], ax=axes[0, i])
    axes[0, i].set_axis_off()
    axes[0, i].set_title('$x^T L x = {:.2f}$'.format(G.dirichlet_energy(x)))

    axes[1, i].plot(G.e, np.abs(x_hat), '.-')
    axes[1, i].set_xticks(range(0, 16, 4))
    axes[1, i].set_xlabel(r'graph frequency $\lambda$')
    axes[1, i].set_ylim(-0.05, 0.95)

axes[1, 0].set_ylabel(r'frequency content $\hat{x}(\lambda)$')

# axes[0, 0].set_title(r'$x$: signal in the vertex domain')
# axes[1, 0].set_title(r'$\hat{x}$: signal in the spectral domain')

fig.tight_layout()
plt.show()
#plt.savefig('heute.svg')