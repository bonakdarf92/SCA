from DarmstadtNetwork import DarmstadtNetwork
import networkx as nx 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import osmnx as ox
import numpy as np
import cvxpy as cp 
from tqdm import tqdm
import gurobipy

# from solver import Solver
# import scipy.stats as ss
# import pygsp as ps 
# plt = settup_pgf()
#mpl.use('pgf')
plt.rcParams.update({'font.size':18})

# Latitude and longitude of cencter of darmstadt city
geo = dict(north=49.874, south=49.8679, west=8.6338, east=8.6517)
D_city = DarmstadtNetwork(geo, "Abgabe")
D_city.load_darmstadt(show=False)
settings = dict(bgcolor="white", equal_aspect=False, node_size=30, node_color='#a142f5', node_edgecolor="black", node_zorder=2, axis_off=False, edge_color="#555555",edge_linewidth=3,edge_alpha=0.7,show=False,close=False,save=False)
xs, ys, ids = D_city.get_ids()
posi = dict(zip(ids, zip(xs, ys)))
D_city.settings = settings

#D_city.plot_map()
#D_city.show_citymap()
Z = np.random.rand(10, 40)
np.savetxt("Z.csv",Z,delimiter=',')
sin1 = np.sin(np.linspace(-np.pi,np.pi,40))
from matplotlib.gridspec import GridSpec
fig = plt.figure(constrained_layout=True)
plt.set_cmap('rainbow')
# gs2 = GridSpec(2,3,figure=fig,left=0.05, right=0.48,wspace=0.03)
# ax0 = fig.add_subplot(gs2[0,0])
# ax1 = fig.add_subplot(gs2[0,1])
# c = ax0.pcolor(sin1*Z)
# ax0.set_title(r'$\mathbf{Y}$')
# ax0.set_axis_off()
# c = ax1.pcolor(sin1*np.ones((10,40)))#, edgecolors='k', linewidths=4)
# ax1.set_title(r'$\mathbf{PQ}$')
# ax1.set_axis_off()
from scipy.sparse import rand as sprand
np.savetxt("PQ.csv",sin1*Z,delimiter=',')
D = np.random.rand(10,60)
S = sprand(60,40,density=0.05)
print(np.shape(np.dot(D,S.todense())))

Z_new = sin1*Z + np.dot(D,S.todense())+np.random.normal(0,0.01,(10,40))# np.dot(D,S)# sin1*Z + np.dot(D,S) + np.random.randn(10,40,0.01)
np.savetxt("Z_new.csv",D,delimiter=',')
np.savetxt("D.csv",D,delimiter=',')
np.savetxt("S.csv",S.todense(),delimiter=',')
gs1 = GridSpec(2,3,figure=fig,left=0.55, right=0.98,wspace=0.03)
ax2 = fig.add_subplot(gs1[0,:-2])
ax3 = fig.add_subplot(gs1[:,2])
c = ax2.pcolor(D)#, edgecolors='k', linewidths=4)
ax2.set_title(r'$\mathbf{D}$')
ax2.set_axis_off()
c = ax3.pcolor(S.todense())
ax3.set_title(r'$\mathbf{S}$')
ax3.set_axis_off()





fig.tight_layout()
plt.show()
#import setBigGraphDarmstadtFinal


