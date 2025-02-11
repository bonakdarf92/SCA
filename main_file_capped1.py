import numpy as np 
from scipy import sparse
#from numba import njit, jit, cuda, float32
import time
#import datetime #from datetime import time 
from Plots.initialPlots import makePlots
#import holoviews as hv 
#from holoviews import opts 
#import holoviews.plotting.mpl
import tqdm


#hv.extension('matplotlib')



"""
N = 200
K = 400
density = 0.01
theta = 0.001
Sample = 1
MaxIter_j = 30 # maximum number of iterations
MaxIter_m = 10 # maximum number of iterations
MaxIter_g = 400 # maximum number of iterations

# the achieved objective value versus the number of iterations
val_j     = np.zeros((Sample, MaxIter_j )) # "_j" stands for the proposed algorithm
val_m     = np.zeros((Sample, MaxIter_m )) # "_m" stands for the majorization-minimization algorithm
val_g     = np.zeros((Sample, MaxIter_g )) # "_g" stands for the GIST algorithm

# the required CPU time versus the number of iterations
time_j    = np.zeros((Sample, MaxIter_j ))
time_m    = np.zeros((Sample, MaxIter_m ))
time_g    = np.zeros((Sample, MaxIter_g ))


# the achieved error versus the number of iterations
error_j   = np.zeros((Sample, MaxIter_j));
error_g   = np.zeros((Sample, MaxIter_g));

theta_vec = theta * np.ones((K,1))
"""



#@jit('Tuple((f8[:,:], f8[:,:], f8, f8[:,:]))(i8,i8,f8)',nopython=True)
def parameter(N, K, density):
    A = np.random.randn(N, K)
    x0 = np.zeros((K,1))
    x0_positions = np.random.choice(np.arange(K), int(K * density), replace = False)
    x0[x0_positions,0] = np.random.normal(0, 1, int(K * density))
    #x_orig = sparse.random(K, 1, density, data_rvs=np.random.randn)
    # TODO Sparsity not working with jit signature 
    sigma2 = 0.0001

    for n in range(N):
        A[n,:] = A[n,:] / np.linalg.norm(A[n,:])
    b = np.dot(A,x0)   + np.sqrt(sigma2) * np.random.randn(N,1)
    #b = A * x_orig + np.sqrt(sigma2) * np.random.randn(N,1)
    mu = 0.1 * np.max(np.abs(np.dot(A.T,b)))
    return A, b, mu, x0 #x_orig


def soft_thresholding(q, t, K):
    '''
    The soft-thresholding function returns the optimal x that minimizes
        min_x 0.5 * x^2 - q * x + t * |x|
    '''
    x = np.maximum(q - t, np.zeros(K)) - np.maximum(-q - t, np.zeros(K));
    return x



# TODO add nopython mode
#@jit('Tuple((f8[:,:],f8[:,:],f8[:,:]))(i8,i8,f8,i8,i8,f8,b1)')   
def GIST_Algo(N, K, density, Sample, MaxIter_g, theta, debug):
    theta_vec = theta * np.ones((K,1))
    time_g = np.zeros((Sample, MaxIter_g))
    val_g = np.zeros((Sample, MaxIter_g))
    error_g = np.zeros((Sample, MaxIter_g))
    for s in range(Sample):
    #for s in tqdm.trange(Sample):
        if debug:
            print("Sample {}".format(s))

        A, b, mu, x0 = parameter(N, K, density)
        
        t_start = time.time_ns()
        mu_vec = mu * np.ones((K,1))
        x_g = np.zeros((K,1))
        residual_g = np.dot(A, x_g) - b
        Gradient_g = (np.dot(residual_g.T, A)).T 
        
        time_g[s,0] = time.time_ns() - t_start
        #test = np.minimum(np.abs(x_g),theta_vec)
        #test2 = 0.5 * np.dot(mu_vec.T,test)
        #val_g[s,0] = 10.0 #float(test2) + float(test)    // TODO check why val_g[access] not working
        val_g[s,0] = 0.5 * np.dot(residual_g.T, residual_g) + np.dot(mu_vec.T, np.minimum(np.abs(x_g), theta_vec))
        #error_g[s,0] = np.linalg.norm(x_g - x0) / np.linalg.norm(x0.toarray())  #sparse version
        error_g[s,0] = np.linalg.norm(x_g - x0) / np.linalg.norm(x0)
        if debug:
            print("Proximal MM Algorithmus: Iteration 0 mit Wert {}".format(val_g[s,1]))

        for t in tqdm.trange(MaxIter_g-1):
            
            t_start = time.time_ns()
            c = 1
            alpha = 0.5
            beta = 2
            while 1:
                u_g = x_g - Gradient_g/c
                x1 = np.sign(u_g) * np.maximum(theta_vec,np.abs(u_g))
                h1 = 0.5 * (x1 - u_g) * (x1 - u_g) + mu*np.minimum(np.abs(x1), theta_vec)
                x2 = np.sign(u_g) * np.minimum(theta_vec, np.maximum(np.zeros((K,1)), np.abs(u_g) - mu_vec/c))
                h2 = 0.5 * (x2 - u_g) * (x2 - u_g) + mu*np.minimum(np.abs(x2), theta_vec)
                
                x_new = x1*(h1<=h2) + x2*(np.ones((K,1)) - (h1<=h2))

                residual_new = np.dot(A, x_new) - b
                val_g_new = 0.5 * np.dot(residual_new.T, residual_new) + np.dot(mu_vec.T, np.minimum(np.abs(x_new), theta_vec))
                temp = (val_g[s,t] - (alpha * c / 2 * np.dot((x_new - x_g).T,(x_new - x_g))))

                if val_g_new <= temp:
                    
                    x_g = x_new
                    val_g[s,t+1] = val_g_new
                    residual_g = residual_new
                    Gradient_g = np.dot(residual_g.T, A).T
                    c = 1
                    break
                else:
                    c *= beta
                    
            time_g[s,t+1] = time.time_ns() - t_start                                      # commented out for @njit
            #error_g[s,t+1] = np.linalg.norm(x_g - x0) / np.linalg.norm(x0.toarray())   #sparse version
            error_g[s,t+1] = np.linalg.norm(x_g - x0) / np.linalg.norm(x0)
            if debug:
                print("Proximal MM Algorithmus: Iteration {} mit Wert {} und Zeit {}".format(t+1, val_g[s,t+1], time_g[s,t+1]))
    return val_g, error_g, time_g
            


def MM_Algo(N, K, density, Sample, MaxIter_m, theta, debug):
    theta_vec = theta * np.ones((K,1))
    time_m = np.zeros((Sample, MaxIter_m))
    val_m = np.zeros((Sample, MaxIter_m))
    for s in range(Sample):
        if debug:
            print("Sample {}".format(s))
        A, b, mu, x0 = parameter(N, K, density)

    t_start = time.time_ns()
    d_Ata = np.reshape(np.sum(np.multiply(A,A),axis=0),(K,1))
    mu_vec = mu * np.ones((K,1))
    mu_vec_norm = mu_vec / d_Ata
    x_m = np.zeros((K,1))
    residual_m = np.dot(A, x_m) - b
    Gradient_m = (np.dot(residual_m.T, A)).T
    time_m[s,0] = time.time_ns() - t_start
    val_m[s,0] = 0.5 * np.dot(residual_m.T, residual_m) + np.dot(mu_vec.T, np.minimum(np.abs(x_m), theta_vec))
    if debug:
            print("Proximal MM Algorithmus: Iteration 0 mit Wert {}".format(val_m[s,1]))
    for t in tqdm.trange(MaxIter_m-1):
        t_start = time.time_ns()
        xi_minus =  mu_vec*(x_m >= theta_vec)  - (x_m <= -theta_vec)
        for k in range(200):
            Bx = soft_thresholding(x_m - (Gradient_m - xi_minus)/d_Ata, mu_vec_norm, K)
            x_diff = Bx - x_m
            Ax_diff = np.dot(A, x_diff)
            stepsize_numerator = -(np.dot(residual_m, Ax_diff) + np.dot(mu_vec, np.absolute(Bx) - np.absolute(x_m)))
            stepsize_denominator = np.dot(Ax_diff, Ax_diff)
            stepsize = np.maximum(np.minimum(stepsize_numerator / stepsize_denominator, 1), 0)

            x_m = x_m + stepsize * x_diff
            residual_m = residual_m + stepsize * Ax_diff
            Gradient_m = (np.dot(residual_m.T, Ax_diff)).T
        time_m[s,t+1] = time.time_ns() - t_start
        val_m[s,t+1] = 0.5 * np.dot(residual_m.T, residual_m) + np.dot(mu_vec.T, np.minimum(np.abs(x_m), theta_vec))
    
    return val_m, time_m





#v,e,t = GIST_Algo(200, 400, 0.01, 10, 400, 0.001, False)
#v1,e1,t1 = GIST_Algo(1000, 1600, 0.01, 3, 400, 0.001, False)
#a,b,c = f(np.random.rand(5),np.random.rand(10,1))
#plot = makePlots((np.linspace(0,np.sum(t),400), np.linspace(0, np.sum(t1), 400)), (v,v1) , legend=["Run 1","Run 2","Run 3"], axes=['s','f'] ,scale="logy", grid=True, saveIt=False)
#plot = makePlots((np.linspace(0,np.sum(t,1),400)), (v) , legend=["Run {}".format(k) for k in range(10)], axes=['s','f'] ,scale="logy", grid=True, saveIt=False)
#v, t = MM_Algo(200, 400, 0.01, 1, 30, 0.001, False)
#plot.show()

#hv.ipython.display(plot)



#makePlots(np.linspace(0,np.sum(t),400), v , legend="Daten", axes=['s','f'] ,scale="logy", grid=True, saveIt=True)
#makePlots(t[0][:],v,scale="logy",grid=True)
#makePlots(np.linspace(0,np.sum(t1),400),v1,scale="logy",grid=True)
#GIST_Algo(N,K,density,time_g,val_g,error_g,MaxIter_g)
