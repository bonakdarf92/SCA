import numpy as np 
#import holoviews as hv 
#from holoviews import opts 
import random

#hv.extension('matplotlib','bokeh')

import pickle
import os
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
import locale
import pandas as pd
import matplotlib as mpl
from datetime import datetime 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
#from stela import stela_lasso
import numpy.matlib as npml


def settup_qd():
    import matplotlib.pyplot as plt1 
    return plt1

def settup_pgf():
    mpl.use("pgf")
    pgf_with_pdflatex = {
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[bitstream-charter]{mathdesign}",
         ]
    }
    #mpl.rcParams.update(pgf_with_pdflatex)
    import matplotlib.pyplot as plt

    mpl.rcParams.update(pgf_with_pdflatex)
    plt.rcParams['font.size'] = 12  # choose font size
    mpl.rc('text', usetex = True)
    plt.rcParams['font.family'] = 'bitstream-charter'
    plt.rcParams['axes.formatter.use_locale'] = True  # use comma (True) or dot (False) separators in graphs
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.loc"] = 'upper right'
    colors = [(0, 0, 0), (185/256, 15/256, 34/256)]  # NTS Colorscheme
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'NTS-Purple'


    # Create the colormap
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    monochrome = (
    cycler('marker', ['', '^', '.', ',']) * cycler('color', ['k','#b90f22']) *cycler('linestyle', ['-','--']))
    window_size = 500
    mf = 600
    return plt



def makePlots(X, Y, legend=None, scale=None, grid=False, axes=['s','f'], saveIt=False, name=None):
    """ This functions reunites and evaluates the inputs X, Y and **kwargs and automates the
    tidious plt routines. Once defined they can be used again and again 

    Arguments:
        X (numpy) array: X-axes if one-dimensional or multiple axes if n-dimensional
        Y (numpy) array: Y-axes if one-dimensional or multple function values if n-dimensional

    optional:
        legend  --  sets the labels for each stream 
        scale   --  loglog, logx, logy or any additional scaled version possible
        grid    --  if set true the grid is shown
        axes    --  uses the array of strings to denote the axes of the plot
        saveIt  --  if set true it saves the plot in pgf format with a given optional Name
        name    --  if name is given the plot is saved with the name if not it uses a timestamp
    """
    if saveIt:                                                      # tree for save flag
        plt = settup_pgf()                                          # run setup routine for saving
        fig = plt.figure()
        dimX = np.shape(X)
        dimY = np.shape(Y)
        s = 0
        if len(dimX) > 1 and len(dimY) > 2:
            s = dimX[0]
        if len(dimX) == 1 and len(dimY) == 2:
            s = dimY[0]
        X_t = np.transpose(X)
        Y_t = np.transpose(Y)


        if axes:                                                    # check axes flag
            if axes[0] == 's' and axes[1] == 'f':                   # check default axes flag
                plt.axes(xlabel="Time in s",ylabel="Objective funtion value f(x)")
            else:
                plt.axes(xlabel=axes[0],ylable=axes[1])

        if s != 1:
            for k in range(s):
                if scale=="logy":                                   # check scale flag - log in y-axes
                    plt.semilogy(X_t[:,k], Y_t[:,0,k], label=legend)
                elif scale=="logx":                                 # check scale flag - log in x-axes
                    plt.semilogx(X_t[:,k], Y_t[:,0,k], label=legend)
                elif scale=="loglog":                               # check scale flag - log log in x/y-axes
                    plt.loglog(X_t[:,k], Y_t[:,0,k], label=legend)
        if s == 1:
            for k in range(s):
                if scale=="logy":                                   # check scale flag - log in y-axes
                    plt.semilogy(X_t, Y_t, label=legend)
                elif scale=="logx":                                 # check scale flag - log in x-axes
                    plt.semilogx(X_t, Y_t, label=legend)
                elif scale=="loglog":                               # check scale flag - log log in x/y-axes
                    plt.loglog(X_t, Y_t, label=legend)
        
        if grid:                                                    # check grid flag
            plt.grid(True)
        if legend:                                                  # check if legend flag
            plt.legend()
        if name:                                                    # check name flag
            plt.savefig(name)
        else:                                                       # if no name is given use timestamp for saving
            now = datetime.now()
            date_time = now.strftime("%Y_%m_%d-%H_%M_%S")
            plt.savefig(date_time)

    else:
        plt1 = settup_qd()
        fig = plt1.figure()
        # TODO check if this part is necessary 
        #k = np.shape(Y)
        #X_t = np.transpose(np.tile(X,(k[0],1)))
        dimX = np.shape(X)
        dimY = np.shape(Y)
        s = 0
        if len(dimX) > 1 and len(dimY) > 2:
            s = dimX[0]
        elif len(dimX) == 1 and len(dimY) == 2:
            s = dimY[1]
        elif len(dimX) == 2 and len(dimY) == 2:
            s = dimX[1]
        else:
            s = 1
        X_t = np.transpose(X)
        Y_t = np.transpose(Y)
        if axes:
            if axes[0] == 's' and axes[1] == 'f':
                plt1.axes(xlabel="Time in s",ylabel="Objective funtion value f(x)")
            else:
                plt1.axes(xlabel=axes[0], ylabel=axes[1])
        
        if s != 1:
            for k in range(s):
                if scale=="logy":
                    if len(legend) > 1:
                        plt1.semilogy(X_t[k,:], Y_t[:,k], label=legend[k])
                    else:
                        plt1.semilogy(X_t[:,k], Y_t[:,0,k], label=legend[k])
                elif scale=="logx":
                    if len(legend) > 1:
                        plt1.semilogx(X_t[k,:], Y_t[:,k], label=legend[k])
                    else:
                        plt1.semilogx(X_t[:,k], Y_t[:,0,k], label=legend[k])
                elif scale=="loglog":
                    if len(legend) > 1:
                        plt1.loglog(X_t[k,:], Y_t[:,k], label=legend[k])
                    else:
                        plt1.loglog(X_t[:,k], Y_t[:,0,k], label=legend[k])
                else:
                    if len(legend) > 1:
                        plt1.plot(X_t,Y_t[k,:],label=legend[k])
                    else:
                        plt1.plot(X_t,Y_t[k,:],label=legend[k])

        elif s == 1:
            for k in range(s):
                if scale=="logy":
                    plt1.semilogy(X_t, Y_t, label=legend)
                elif scale=="logx":
                    plt1.semilogx(X_t, Y_t, label=legend)
                elif scale=="loglog":
                    plt1.loglog(X_t, Y_t, label=legend)
            
        
        if grid:
            plt1.grid(True)
        #else:
        #    plt1.plot(X_t, Y_t, label=legend)

        if legend:
            plt1.legend()
        #plt1.show()
        return plt1
   

def signalPoint(Sensor,show=False,title=None):
    plt1 = settup_qd()
    fig = plt1.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    linestyle_tuple = [
     (0, (1, 1)),
     #(0, (1, 1)),
     #(0, (5, 10)),
     (0, (5, 5)),
     (0, (5, 1)),
     #(0, (3, 5, 1, 5)),
     (0, (3, 1, 1, 1)),
     #(0, (3, 5, 1, 5, 1, 5)),
     (0, (3, 1, 1, 1, 1, 1))]

    n,m = np.shape(Sensor)
    plt1.gca().set_prop_cycle(plt1.cycler('color', plt1.cm.rainbow(np.linspace(0,1, m))))

    for k in range(m):
        if k < m/2:
            ax.plot(range(n),Sensor[:,k],zs=k,zdir='y',linestyle=random.choice(linestyle_tuple))
        else:
            ax2.plot(range(n),Sensor[:,k],zs=k,zdir='y',linestyle=random.choice(linestyle_tuple))
    if title:
        plt1.title(title)
    if show:
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d-%H_%M_%S")
        plt1.savefig(date_time)
    else:
        return plt1


def generateNumbers(n,k,kind=None,specs=None,seed=None):
    if k:
        if kind=='zeros':
            return np.zeros((n,k))
        elif kind=='rand':
            if seed:
                np.random.seed(seed)
            return np.random.randn(n,k)
        elif kind=='ones':
            return np.ones((n,k))
        elif kind=='gauss':
            if seed:
                np.random.seed(seed)
            return np.random.normal(0,specs,(n,k))
        elif kind=='unique':
            pass
    else:
        if kind=='zeros':
            return np.zeros(n)
        elif kind=='rand':
            if seed:
                np.random.seed(seed)
            return np.random.randn(n)
        elif kind=='ones':
            return np.ones(n)
        elif kind=='gauss':
            if seed:
                np.random.seed(seed)
            return np.random.normal(0,specs,n)
        elif kind=='unique':
            pass 

def vectorType(x):
    return {
            'zeros': np.zeros(n,k),
            'rand':  np.random.randn(n,k),
            'ones':  3,
            'gaus':  4,
            'unique':5,
        }[x]


def alpine01(x,y):
    return np.absolute(x * np.sin(x) + 0.1*x) + np.absolute(y*np.sin(y) + 0.1*y)

"""
x = np.arange(-10,10,0.1)
y = np.arange(-10,10,0.1)
xx,yy = np.meshgrid(x,y,sparse=False)
z = alpine01(xx,yy)
x0 = 3.6*np.ones(2)

z_vec = z.reshape((40000,1), order="F")
#a1 = np.transpose(npml.repmat(x,1,20))
#a2 = np.reshape(np.repeat(y,20,axis=0),(400,1))
a1 = np.reshape(np.repeat(x,200,axis=0),(40000,1))
a2 = np.transpose(npml.repmat(y,1,200))
A = np.hstack((a1,a2))
measurement = np.reshape(z_vec,(40000,))# + np.random.normal(0,1,40000)
mu = 0.001*np.linalg.norm(np.dot(np.reshape(z_vec,(40000,)), A),np.inf)
print(mu)
objval, x, error = stela_lasso(A, measurement, mu, 5000)
print(x)
plt = settup_qd()
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(xx,yy,z,cmap=cm.coolwarm,linewidth=0,antialiased=False)

plt.show()
"""