import numpy as np
import matplotlib.pyplot as plt
from stela import stela_lasso, soft_thresholding

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
plt.close()
D1 = D_g.D.toarray() 
L = D_g.L.toarray()
#A = np.random.normal(0,0.1,(1000,100))
#N = 1000
#K = 100
density = 0.5
#A = np.concatenate((L,L),axis=0)  # doppelt Eintrag
A = L
N,K = A.shape
x0 = np.zeros(K)
x0_positions = np.random.choice(np.arange(K),int(K*density),replace=False)

x0[x0_positions] = 5*np.random.normal(0,1,int(K*density))  # sparse signal
#x0 = np.random.normal(0,1,int(K))
#x_up = np.concatenate((x0,x0),axis=0)   # doppelt Eintrag

#D_g.plot()
sigma = 0.001
v = np.random.normal(0,sigma,N)
#A = np.concatenate((A,A),axis=1)       # doppelt Eintrag
#v = np.concatenate((v,v),axis=0)
rs = np.random.RandomState(42)
M = (rs.rand(D_g.N,D_g.N) > 0.1).astype(float)
y = np.dot(A,x0) + v
mask = np.ones(len(x0),np.bool)
mask[x0_positions] = False
selec = np.random.choice(x0[mask],int(K*0.5),replace=False).astype(int)
y[selec] = 0
#y = np.dot(M,x0) + v

mu = 0.0001 * np.linalg.norm(np.dot(y,A),np.inf)
theta = 50
maxiter = 25000
objval, x, error = stela_cappedL1(A,y,mu,theta,maxiter)

'''plot output'''
'''compare the original signal and the estimated signal'''
plt.plot(np.linspace(1, K, K), x0, 'bx', label = "original signal")
plt.plot(np.linspace(1, K, K), x, 'ro',mfc='none', label = "estimated signal")
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
