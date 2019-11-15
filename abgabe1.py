from DarmstadtNetwork import DarmstadtNetwork
import networkx as nx 
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
import cvxpy as cp 
#from cvxopt import lapack, solvers, matrix, spdiag, log, div, normal 
import gurobipy

# Load small darmstadt view
geo = dict(north=49.874,south=49.8679,west=8.6338,east=8.6517)
D_city = DarmstadtNetwork(geo,"Abgabe")
D_city.load_darmstadt(show=False)
fig,ax = ox.plot_graph(D_city.Graph,**D_city.settings)
xs,ys,ids = D_city.get_ids()
posi = dict(zip(ids,zip(xs,ys)))

NX = False
# Draw the topology of darmstadt with streetnames
hin = [k for k in D_city.Graph.edges.data('name')]
edgepos = [hin[k][0:2] for k in range(len(hin))]
edgename = [hin[k][2] for k in range(len(hin))]
if NX:
    nx.draw_networkx(D_city.Graph,pos=posi,with_labels=False,node_size=20,ax=ax)
    nx.draw_networkx_edge_labels(D_city.Graph,pos=posi,edge_labels=dict(zip(edgepos,edgename)))
    plt.tight_layout()
    plt.show()
# Draw with weights
strecke = [k for k in D_city.Graph.edges.data('length')]
edgeposs = [strecke[k][0:2] for k in range(len(strecke))]
edgestrecke = [strecke[k][2] for k in range(len(strecke))]
if NX:
    fig,ax = ox.plot_graph(D_city.Graph,**D_city.settings)
    nx.draw_networkx(D_city.Graph, pos=posi,with_labels=False,node_size=20,ax=ax)
    nx.draw_networkx_edge_labels(D_city.Graph,pos=posi,edge_labels=dict(zip(edgeposs,edgestrecke)))
    plt.tight_layout()
    plt.show()

if NX:
    fig,ax = ox.plot_graph(D_city.Graph, **D_city.settings)
    nx.draw_networkx_nodes(D_city.Graph, pos=posi, nodelist=D_city.Graph.nodes(), with_labels=False, node_size=30,ax=ax)
    nx.draw_networkx_labels(D_city.Graph, pos=posi, labels=dict(zip(D_city.Graph.nodes(),range(D_city.Graph.number_of_nodes()))))
    plt.tight_layout()
    plt.show()
bla = nx.to_directed(D_city.Graph)

for k in range(len(strecke)):
    list_strecke = list(strecke[k][:])
    list_strecke[2] = round(np.exp(-(np.square(list_strecke[2]))/(2*100**2)))
    bla.add_weighted_edges_from([tuple(list_strecke)])

A = D_city.sparse_adj
A2 = D_city.remove_diagsLoops(direction="directed")
A3 = nx.convert_matrix.to_scipy_sparse_matrix(bla)

import pygsp as ps 
G = ps.graphs.Graph(A)
G.set_coordinates([[v,z] for v,z in zip(xs,ys)])
G2 = ps.graphs.Graph(A3)
G2.set_coordinates([[v,z] for v,z in zip(xs,ys)])

#_ = ax1[0].spy(G.W,markersize=2)
G.compute_laplacian('combinatorial')
G.compute_fourier_basis()
G.compute_differential_operator()
G2.compute_laplacian('combinatorial')
G2.compute_fourier_basis()
G2.compute_differential_operator()
rs = np.random.RandomState(42)
"""
filt = lambda x: 1 / (1 + 10*x)
filt = ps.filters.Filter(G2,filt)
signal_tik = filt.analyze(rs.normal(size=G2.N))
mask = rs.uniform(0,1,G2.N) > 0.7
measure = signal_tik.copy()
measure[~mask] = np.nan 
recovery = ps.learning.regression_tikhonov(G2,measure,mask,tau=0.01)
fig3,ax3 = plt.subplots(1,3,sharey=True,figsize=(10,3))
limits = [signal_tik.min(),signal_tik.max()]

_ = G2.plot_signal(signal_tik,ax=ax3[0],limits=limits,title='Ground Truth')
_ = G2.plot_signal(measure,ax=ax3[1],limits=limits,title='Measurement')
_ = G2.plot_signal(recovery,ax=ax3[2],limits=limits,title='Recovery ')
_ = fig3.tight_layout()
plt.show()
plt.plot(signal_tik-recovery)
plt.show()
"""
plt.show()

def gen_signal(sigma=0.1):
    sources_path = [20,41,74,6,16,45,68,57,15,30,11,23,43,24]#(rs.rand(G2.n_vertices) > 0.9).astype(bool)
    signal = np.zeros(G2.n_vertices)
    signal[sources_path] = 1
    noisy = signal + np.random.normal(0, 0.4, G2.n_vertices)
    return noisy

def mse(x, t):
    return cp.norm2(x - t)**2


def cut_based(y, Graph, lambd):
    x = cp.Variable(Graph.n_vertices, boolean=True)
    object_cut = cp.Minimize(cp.sum_squares(y - x) + lambd * (cp.norm1(Graph.A @ x) - cp.quad_form(x, Graph.A)))
    problem_cut = cp.Problem(object_cut)
    problem_cut.solve(solver=cp.GUROBI)
    return x, lambd
    
def path_based1(y, Graph, tp):
    

x = cp.Variable(G2.n_vertices, boolean=True)
obje = cp.Minimize(cp.sum_squares(noisy-x) + 0.5 * cp.quad_form(x,G2.L))
problem = cp.Problem(obje)
problem.solve(solver=cp.GUROBI,verbose=True)

x1 = cp.Variable(G2.n_vertices, nonneg=True)
constr = [cp.norm(A3@x1, 'inf') <= 2, x1 <= 1]
obje_path = cp.Minimize(cp.sum_squares(noisy - x1))
prog = cp.Problem(obje_path, constr)
prog.solve(solver=cp.GUROBI)

x2 = cp.Variable(G2.n_vertices, nonneg=True)
constr2 = [cp.norm(A3@x2, 'inf') <= 2, x2 <= 1]
obj_far = cp.Minimize(cp.sum_squares(noisy - x2) + (2*np.max(signal) - 1)*cp.sum(x2))
prog2 = cp.Problem(obj_far, constr2)
prog2.solve(solver=cp.GUROBI)

#x2.value[x2.value <= 0.1] = 0
#x2.value[x2.value >= 0.1] = 1

x3 = cp.Variable(G2.n_vertices, nonneg=True)
resid = x3 - 1
f1 = cp.norm(A3@x3, 'inf') - 2 #+ cp.sum_squares(resid)
theta = cp.Parameter(nonneg=True)
theta.value = 0.01
aug_lagr = cp.sum_squares(noisy - x3) + theta * f1 
analy = []
for eps in range(10):
    analy.append(theta.value)
    obj_far = cp.Problem(cp.Minimize(aug_lagr)).solve(solver=cp.GUROBI)
    theta.value += theta.value     
    #prog2.solve(solver=cp.GUROBI)

plt.scatter(range(75),x2.value,label='t-0.4',marker='x')
plt.scatter(range(75),signal,label='Original',marker='o',edgecolor='k',facecolor='none')
plt.scatter(range(75),x1.value,label='Paper',marker='v')
plt.legend()
plt.show()

plt.plot(analy)
plt.show()

fig1,ax1 = plt.subplots(2,1,figsize=(12,8))
plt.set_cmap('seismic')
plt.tight_layout()
_,_,we = G2.get_edge_list()
times = [0]#, 5, 20]
for i, t in enumerate(times):
    g = ps.filters.Heat(G2,scale=t,normalize=False)
    #title = r'$\hat{{f}}({0}) = g_{{1,{0}}} \odot \hat{{f}}(0)$'.format(t)
    title = r"Noisy Signal$ y = f(x) + \sigma$"
    #g.plot(alpha=1,ax=ax1[0,i],title=title)
    #g.plot(alpha=1,ax=ax1[0],title=title)
    G2.plot(noisy,edges=True,edge_width=we, highlight=sources_path,ax=ax1[0],title=title)
    #ax1[0,i].set_xlabel(r'$\lambda$')
    #ax1[0].set_xlabel(r'$\y = Lx + \sigma $')
    if i > 0:
        ax1[0,i].set_ylabel('')
    y = g.filter(signal)
    #line, = ax1[0,i].plot(G2.e,G2.gft(y))
    #line, = ax1[0].plot(G2.e,G2.gft(y))
    #labels = [r'$\hat{{f}}({})$'.format(t), r'$g_{{1,{}}}$'.format(t)]
    #ax1[0, i].legend([line, ax1[0, i].lines[-3]], labels, loc='lower right')
    #ax1[0].legend([line, ax1[0].lines[-3]], labels, loc='lower right')
    #G2.plot(y, edges=True,edge_width=we, highlight=sources, ax=ax1[1, i], title=r'$f({})$'.format(t))
    #G2.plot(y, edges=True,edge_width=we, highlight=sources, ax=ax1[1], title=r'$f({})$'.format(t))
    G2.plot(x.value,edges=True, edge_width=we, highlight=[i for i,k in enumerate(x.value) if k > 0.9 ], ax=ax1[1])
    #print(np.sum(y))
    #ax1[1,i].set_aspect('equal','datalim')
    #ax1[1,i].margins(x=-0.3,y=-0.49)
    #ax1[1,i].set_axis_off()
    ax1[1].set_aspect('equal','datalim')
    ax1[1].margins(x=-0.2,y=-0.4)
    ax1[1].set_axis_off()
    ax1[0].set_aspect('equal','datalim')
    ax1[0].margins(x=-0.2,y=-0.4)
    ax1[0].set_axis_off()



plt.show()

from Plots import initialPlots

sensor_data = np.load('.\\Darmstadt_verkehr\\SensorData_{}.npz'.format('Sensor_Small_View'),allow_pickle=True)['arr_0'].reshape((1,))[0]

a003 = np.nansum(sensor_data['A003']['signals'][:,0:11],axis=1) + sensor_data['A003']['signals'][:,22]
a004 = np.nansum(sensor_data['A004']['signals'][:,0:11],axis=1)
a005 = np.nansum(sensor_data['A005']['signals'][:,0:5],axis=1)
a006 = np.nansum(sensor_data['A006']['signals'][:,[0,1,2,6,14]],axis=1)
a007 = np.nansum(sensor_data['A007']['signals'][:,0:3],axis=1)
a022 = np.nansum(sensor_data['A022']['signals'][:,0:9],axis=1)
a023 = np.nansum(sensor_data['A023']['signals'][:,0:12],axis=1)
a028 = np.nansum(sensor_data['A028']['signals'][:,13:19],axis=1)
a029 = np.nansum(sensor_data['A029']['signals'][:,0:1],axis=1)
a030 = np.nansum(sensor_data['A030']['signals'][:,16:18],axis=1)
a045 = (np.nansum(sensor_data['A045']['signals'][:,0:4],axis=1) 
        + np.nansum(sensor_data['A045']['signals'][:,6:7],axis=1)  
        + np.nansum(sensor_data['A045']['signals'][:,9:12],axis=1))
a102 = np.nansum(sensor_data['A102']['signals'][:,0:1],axis=1)
a104 = np.nansum(sensor_data['A104']['signals'][:,[2,3,4,5,14,15,18,19]],axis=1)
#a139 = np.nansum(sensor_data['A139']['signals'][:,0:12]) + sensor_data['A003']['signals'][:,22]

stack = np.array((a003,a004,a005,a006,a007,a022,a023,a028,a030,a045,a102,a104))
plt1 = initialPlots.signalPoint(stack.T,show=False,title="A4")
plt1.xticks((0, 60, 120, 180, 240, 300, 360, 420, 480, 540,
            600, 660, 720, 780, 840, 900, 960, 1020, 1080,
            1140, 1200, 1260, 1320, 1380, 1440),
            ("0","1","2","3","4","5","6","7",
            "8","9","10","11","12","13","14",
            "15","16","17","18","19","20","21","22","23"))
plt1.xlabel("Uhrzeit")
plt1.show()
#initialPlots.makePlots(range(1440),sensor_data,grid=True)