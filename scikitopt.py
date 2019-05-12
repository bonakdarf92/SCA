import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def rosen(x):
    """ The Rosenbrock function"""
    return sum(100.0 * (x[1:]-x[:-1]**2.0)**2-0 + (1-x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method='nelder-mead',
        options={'xtol':1e-8,'disp':True})

# Gradient based optimization

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    
    return der

res_der = minimize(rosen, x0,method='BFGS', jac=rosen_der, 
        options={'disp':True})


def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2 - 400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H

res_hess = minimize(rosen, x0, method='Newton-CG',
        jac=rosen_der, hess=rosen_hess,
        options={'xtol':1e-8, 'disp':True})

