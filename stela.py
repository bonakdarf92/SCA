import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import huber as hb


def soft_thresholding(q, t, K):
    """
    The soft-thresholding function returns the optimal x that minimizes
        min_x 0.5 * x^2 - q * x + t * |x|
    """
    x = np.maximum(q - t, np.zeros(K)) - np.maximum(-q - t, np.zeros(K))
    return x


def soft_thresholding_pos(q, t, K):
    """
    The soft-thresholding function returns the optimal x that minimizes
        min_x 0.5 * x^2 - q * x + t * |x|
    """
    x = np.maximum(q - t, np.zeros(K))
    return x

def huber_thresholing(q,l,delta=1):
    return delta* np.divide((np.divide(q, delta) + l*np.maximum(np.abs(np.divide(q,delta)) - l*np.ones_like(q) - np.ones_like(q),np.zeros_like(q))), l*np.ones_like(q) + np.ones_like(q))


def capped_thresholding(q,t):
    return np.maximum(q - t*np.ones_like(q) + np.minimum(q, t*np.ones_like(q)), np.zeros_like(q))


def huber_loss(q,delta=1):
    if np.less_equal(np.abs(q), delta*np.ones_like(q)):
        return 0.5*q*q
    else:
        return delta * (np.abs(q) - 0.5*delta)


def stela_lasso(A, y, mu, MaxIter=1000,mode="ls",verbosity=True):
    """
    STELA algorithm solves the following optimization problem:
        min_x 0.5*||y - A * x||^2 + mu * ||x||_1

    Reference:
        Sec. IV-C of [Y. Yang, and M. Pesavento, "A unified successive pseudoconvex approximation framework", IEEE Transactions on Signal Processing, 2017]
    Input Parameters:
        A :      N * K matrix,  dictionary
        y :      N * 1 vector,  noisy observation
        mu:      positive scalar, regularization gain
        MaxIter: (optional) maximum number of iterations, default = 1000

    Definitions:
        N : the number of measurements
        K : the number of features
        f(x) = 0.5 * ||y - A * x||^2
        g(x) = mu * ||x||_1

    Output Parameters:
        x:      K * 1 vector, the optimal variable that minimizes {f(x) + g(x)}
        objval: objective function value = f + g
        error:  specifies the solution precision (a smaller error implies a better solution), defined in (53) of the reference

    """
    if mu <= 0:
        print('mu must be positive!')
        return
    elif A.shape[0] != y.shape[0]:
        print('The number of rows in A must be equal to the dimension of y!')
        return

    '''precomputation'''
    K = A.shape[1]
    test = np.multiply(A,A)
    AtA_diag = np.sum(np.multiply(A, A), axis=0)  # diagonal elements of A'*A
    mu_vec = mu * np.ones(K)
    mu_vec_normalized = np.divide(mu_vec, AtA_diag)

    '''initialization'''
    x = np.zeros(K)  # x is initialized as all zeros
    objval = np.zeros(MaxIter + 1)  # objvective function value vs number of iterations
    error = np.zeros(MaxIter + 1)  # solution precision vs number of iterations
    CPU_time = np.zeros(MaxIter + 1)  # CPU time (cumulative with respect to iterations)

    '''The 0-th iteration'''
    CPU_time[0] = time.time()
    residual = np.dot(A, x) - y  # residual = A * x - y
    #print(residual.shape)
    f_gradient = np.dot(residual, A)  # gradient of f
    #print(f_gradient.shape)
    CPU_time[0] = time.time() - CPU_time[0]

    f = 1 / 2 * np.dot(residual, residual)
    g = mu * np.linalg.norm(x, 1)
    objval[0] = f + g
    error[0] = np.linalg.norm(
        np.absolute(f_gradient - np.minimum(np.maximum(f_gradient - x, -mu * np.ones(K)), mu * np.ones(K))),
        np.inf)  # cf. (53) of reference

    if verbosity:
        '''print initial results'''
        IterationOutput = "{0:9}|{1:10}|{2:15}|{3:15}|{4:15}"
        print(IterationOutput.format("Iteration", "stepsize", "objval", "error", "CPU time"))
        print(
            IterationOutput.format(0, 'N/A', format(objval[0], '.7f'), format(error[0], '.7f'), format(CPU_time[0], '.7f')))

    '''formal iterations'''
    for t in range(0, MaxIter):
        CPU_time[t + 1] = time.time()

        '''approximate problem, cf. (49) of reference'''
        #Bx = soft_thresholding(x - np.divide(f_gradient, AtA_diag), mu_vec_normalized, K)
        if mode == "ls":
            Bx = soft_thresholding(x - np.divide(f_gradient, AtA_diag), mu_vec_normalized, K)
        elif mode == "huber":
            Bx = huber_thresholing(x - np.divide(f_gradient, AtA_diag),mu,0.05)
        x_dif = Bx - x
        Ax_dif = np.dot(A, x_dif)  # A * (Bx - x)

        '''stepsize, cf. (50) of reference'''
        stepsize_numerator = -(np.dot(residual, Ax_dif) + np.dot(mu_vec, np.absolute(Bx) - np.absolute(x)))
        stepsize_denominator = np.dot(Ax_dif, Ax_dif)
        stepsize = np.maximum(np.minimum(stepsize_numerator / stepsize_denominator, 1), 0)

        '''variable update'''
        x = x + stepsize * x_dif
        residual = residual + stepsize * Ax_dif
        f_gradient = np.dot(residual, A)

        CPU_time[t + 1] = time.time() - CPU_time[t + 1] + CPU_time[t]
        if mode == "ls":
            f = 1 / 2 * np.dot(residual, residual)
        elif mode == "huber":
            f = 1/2 * np.sum(hb(0.05,residual))

        g = mu * np.linalg.norm(x, 1)
        objval[t + 1] = f + g
        if mode == "huber":
            error[t + 1] = np.abs(stepsize_numerator)
        else:
            #error[t + 1] = np.linalg.norm(np.absolute(f_gradient - np.minimum(np.maximum(f_gradient - x, -mu_vec), mu_vec)),
            #                          np.inf)
            error[t+1] = np.abs( np.dot(x_dif.T,f_gradient) + mu* np.linalg.norm(Bx,1) - g)
        
        if verbosity:
            '''print intermediate results'''
            print(IterationOutput.format(t + 1, format(stepsize, '.7f'), format(objval[t + 1], '.7f'),
                                        format(error[t + 1], '.7f'), format(CPU_time[t + 1], '.7f')))

        '''check stop criterion'''
        if error[t + 1] < 1e-6:
            objval = objval[0: t + 2]
            CPU_time = CPU_time[0: t + 2]
            error = error[0: t + 2]
            if verbosity:
                print('Status: successful')
            break

        if t == MaxIter - 1:
            if verbosity:
                print('Status: desired precision is not achieved. More iterations are needed.')


    return objval, x, error


def stela_cappedL1(A, y, mu, theta, maxiter=1000):
    '''
    STELA algorithm solves the following optimization problem:
        min_x 0.5*||y - A * x||^2 + mu * ||x||_1

    Reference:
        Sec. IV-C of [Y. Yang, and M. Pesavento, "Successive convex optimization algorithms for sparse signal estimation with nonconvex regularitaions", IEEE Transactions on Signal Processing, 2017]
    Input Parameters:
        A :      N * K matrix,  dictionary
        y :      N * 1 vector,  noisy observation
        mu:      positive scalar, regularization gain
        MaxIter: (optional) maximum number of iterations, default = 1000

    Definitions:
        N : the number of measurements
        K : the number of features
        f(x) = 0.5 * ||y - A * x||^2
        g+(x) = mu * ||x||_1
        g-(x) = mu * ||x||_1 - mu ||min(x,theta)||_1

    Output Parameters:
        x:      K * 1 vector, the optimal variable that minimizes {f(x) + g(x)}
        objval: objective function value = f + g
        error:  specifies the solution precision (a smaller error implies a better solution), defined in (53) of the reference

    '''
    if mu <= 0:
        print('mu must be positive!')
        return
    elif A.shape[0] != y.shape[0]:
        print('The number of rows in A must be equal to the dimension of y!')
        return

    '''precomputation'''
    K = A.shape[1]
    AtA_diag = np.diag(np.dot(A.T,A))#np.sum(np.multiply(A, A), axis=0)  # diagonal elements of A'*A
    mu_vec = mu * np.ones(K)
    mu_vec_normalized = np.divide(mu_vec, AtA_diag)
    theta_vec = theta * np.ones(K)

    '''initialization'''
    x = np.zeros(K)  # x is initialized as all zeros
    objval = np.zeros(maxiter + 1)  # objvective function value vs number of iterations
    error = np.zeros(maxiter + 1)  # solution precision vs number of iterations
    CPU_time = np.zeros(maxiter + 1)  # CPU time (cumulative with respect to iterations)

    '''The 0-th iteration'''
    CPU_time[0] = time.time()
    residual = np.dot(A, x) - y  # residual = A * x - y
    print(residual.shape)
    f_gradient = np.dot(residual, A)  # gradient of f
    print(f_gradient.shape)
    CPU_time[0] = time.time() - CPU_time[0]

    f = 1 / 2 * np.dot(residual, residual)
    g = np.dot(mu_vec, np.minimum(np.absolute(x),theta_vec))
    objval[0] = f + g
    error[0] = np.linalg.norm(
        np.absolute(f_gradient - np.minimum(np.maximum(f_gradient - x, -mu * np.ones(K)), mu * np.ones(K))),
        np.inf)  # cf. (53) of reference

    '''print initial results'''
    IterationOutput = "{0:9}|{1:10}|{2:15}|{3:15}|{4:15}"
    print(IterationOutput.format("Iteration", "stepsize", "objval", "error", "CPU time"))
    print(IterationOutput.format(0, 'N/A', format(objval[0], '.7f'), format(error[0], '.8f'), format(CPU_time[0], '.4f')))

    '''formal iterations'''
    for t in range(0, maxiter):
        CPU_time[t + 1] = time.time()

        xi_minus =  mu_vec*(x >= theta_vec)  - (x <= -theta_vec)

        '''approximate problem, cf. (49) of reference'''
        Bx = soft_thresholding(x - np.divide(f_gradient - xi_minus, AtA_diag), mu_vec_normalized, K)

        x_dif = Bx - x
        Ax_dif = np.dot(A, x_dif)  # A * (Bx - x)

        '''stepsize, cf. (50) of reference'''
        #stepsize_numerator = -(np.dot(residual, Ax_dif) + np.dot(mu_vec, np.absolute(Bx) - np.absolute(x)) - np.dot(xi_minus,x_dif))
        stepsize_numerator = -(np.dot(residual, Ax_dif) + (np.dot(mu_vec, np.absolute(Bx) - np.absolute(x)) - np.dot(xi_minus,x_dif)))
        stepsize_denominator = np.dot(Ax_dif, Ax_dif)
        #stepsize = np.maximum(np.minimum(stepsize_numerator / stepsize_denominator, 1), 0)
        stepsize = np.minimum(stepsize_numerator / stepsize_denominator, 1)
        '''variable update'''
        x = x + stepsize * x_dif

        residual = residual + stepsize * Ax_dif
        f_gradient = np.dot(residual, A)

        CPU_time[t + 1] = time.time() - CPU_time[t + 1] + CPU_time[t]

        f = 1 / 2 * np.dot(residual, residual)
        g = mu * np.linalg.norm(x, 1)
        objval[t + 1] = f + g
        #error[t + 1] = np.linalg.norm(np.absolute(f_gradient - np.minimum(np.maximum(f_gradient - x, -mu_vec), mu_vec)),
        #                              np.inf)
        error[t + 1] = np.absolute(np.dot(f_gradient - xi_minus,x_dif) + mu*(np.linalg.norm(Bx,1) - np.linalg.norm(x,1)))
        '''print intermediate results'''
        print(IterationOutput.format(t + 1, format(stepsize, '.4f'), format(objval[t + 1], '.7f'),
                                     format(error[t + 1], '.8f'), format(CPU_time[t + 1], '.4f')))

        '''check stop criterion'''
        if error[t + 1] < 1e-6:
            objval = objval[0: t + 2]
            CPU_time = CPU_time[0: t + 2]
            error = error[0: t + 2]
            print('Status: successful')
            break

        if t == maxiter - 1:
            print('Status: desired precision is not achieved. More iterations are needed.')
    
    return objval, x, error

A = np.random.normal(0,0.1,(100,1000))
x0 = np.zeros(1000)
x0_pos = np.random.choice(np.arange(1000),int(1000*0.01),replace=False)
x0[x0_pos] = np.random.normal(0,1,int(1000*0.01))
sigma = 0.01
v = np.random.normal(0,sigma,100)
y = np.dot(A,x0) + v 
mu = 0.1*np.linalg.norm(np.dot(y,A),np.inf)

objval_ls,x_ls,err_ls = stela_lasso(A,y,mu,1000,mode="ls")
objval_hb,x_hb,err_hb = stela_lasso(A,y,mu,1000,mode="huber")



plt.plot(np.linspace(1, 1000, 1000), x0, 'bx', label = "original signal")
plt.plot(np.linspace(1, 1000, 1000), x_ls, 'ro', label = "estimated signal ls")
plt.plot(np.linspace(1, 1000, 1000), x_hb, 'g-.', label = "estimated signal hb")
plt.legend(bbox_to_anchor = (1.05, 1), loc = 1, borderaxespad = 0.)
plt.xlabel("index")
plt.ylabel("coefficient")
plt.show()

plt.plot(np.linspace(0, objval_ls.size-1, objval_ls.size), objval_ls, 'r-x',label="Least")
plt.plot(np.linspace(0, objval_hb.size-1, objval_hb.size), objval_hb, 'g-.',label="Huber")
plt.legend(bbox_to_anchor = (1.05, 1), loc = 1, borderaxespad = 0.)
plt.xlabel("number of iterations")
plt.ylabel("objective function value")
plt.show()

plt.plot(np.linspace(0, err_ls.size-1, err_ls.size), err_ls, 'r-x',label="Least")
plt.plot(np.linspace(0, err_hb.size-1, err_hb.size), err_hb, 'g-.',label="Huber")
plt.legend(bbox_to_anchor = (1.05, 1), loc = 1, borderaxespad = 0.)
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.yscale('log')
plt.show()