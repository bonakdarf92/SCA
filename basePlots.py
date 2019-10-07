import pickle
import numpy as np
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
print(sys.path)
from stela import stela_lasso
import numpy.matlib as npml

#os.path.join('')
def settup_qd():
    import matplotlib.pyplot as plt 
    return plt

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

    import matplotlib.pyplot as plt

    mpl.rcParams.update(pgf_with_pdflatex)
    plt.rcParams['font.size'] = 12  # choose font size
    mpl.rc('text', usetex = True)
    plt.rcParams['font.family'] = 'bitstream-charter'
    plt.rcParams['axes.formatter.use_locale'] = True  # use comma (True) or dot (False) separators in graphs
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.loc"] = 'lower right'
    colors = [(0, 0, 0), (185/256, 15/256, 34/256)]  # PTW Colorscheme
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'NTS-Purple'


    # Create the colormap
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    monochrome = (
    cycler('marker', ['', '^', '.', ',']) * cycler('color', ['k','#b90f22']) *cycler('linestyle', ['-','--']))
    window_size = 500
    mf = 600
    return plt



def makePlots(X, Y, legend=None, scale=None, saveIt=False,name=None):
    if saveIt:
        plt = settup_pgf()
        k = np.shape(Y)
        X_t = np.transpose(np.tile(X,(k[0],1)))
        Y_t = np.transpose(Y)
        plt.plot(X_t, Y_t)
        if name:
            plt.savefig(name)
        else:
            now = datetime.now()
            date_time = now.strftime("%Y_%m_%d-%H:%M:%S")
            plt.savefig(date_time)
    else:
        #import matplotlib.pyplot as plt
        plt = settup_qd()
        k = np.shape(Y)
        X_t = np.transpose(np.tile(X,(k[0],1)))
        Y_t = np.transpose(Y)
        plt.plot(X_t, Y_t)
        plt.show()
   
def generateNumbers(n,k,kind=None,specs=None,seed=None):
    if k:
        if kind=='zeros':
            return np.zeros(n,k)
        elif kind=='rand':
            if seed:
                np.random.seed(seed)
            return np.random.randn(n,k)
        elif kind=='ones':
            return np.ones(n,k)
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

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# surf = ax.plot_surface(x,y,z)
# plt.show()

A = generateNumbers(10,k=10,kind='rand')
x = np.linspace(-10,10)
y0 = np.sin(x)
y1 = np.tanh(x)
y2 = np.abs(x)
y3 = np.cosh(x)**2 + np.sin(x) 
X = [x, x, x]
Y = [y0, y1, y2]
makePlots(X,Y)
#makePlots(x, Y,saveIt=True,name=None)


"""
mf = 800
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4,ncols=1, figsize=(12, 6), dpi=130)
ax0.set_prop_cycle(monochrome)
ax0.set_ylabel('Schaltvorgänge')
ax0.set_title('PPO, MLP, mit Prognose', loc='left', fontsize='x-small')
ax1.set_prop_cycle(monochrome)
ax1.set_title('PPO, MLP, ohne Prognose', loc='left', fontsize='x-small')
ax2.set_prop_cycle(monochrome)
ax2.set_title('PPO, LSTM, mit Prognose', loc='left', fontsize='x-small')
ax3.set_prop_cycle(monochrome)
ax3.set_title('TRPO, MLP, mit Prognose', loc='left', fontsize='x-small')
ax0.plot(switch_clean_test1[:-50], markevery=mf)
ax0.legend(['chP','Con Boil', 'Im Heat', 'Bat'],bbox_to_anchor=(0.3, 1.4), loc='upper left', ncol=4, fontsize='x-small', handlelength=3.0)
ax1.plot(switch_clean_test2[:-50], markevery=mf)
ax2.plot(switch_clean_test3[:-50], markevery=mf)
ax3.plot(switch_clean_test4[:-50], markevery=mf)
ax.set_xlabel('Anzahl der Episoden (1 Episode = 3 Tage')
fig.subplots_adjust(hspace=0.7)
plt.savefig(os.path.join(os.getcwd(), 'training_data\\new_runs\\comparison_switches.pgf'))
"""

print("ok")

