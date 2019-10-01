from DarmstadtNetwork import DarmstadtNetwork
import networkx as nx 
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
geo = dict(north=49.874,south=49.8679,west=8.6338,east=8.6517)
D_city = DarmstadtNetwork(geo,"Abgabe")
D_city.load_darmstadt(show=False)
fig,ax = ox.plot_graph(D_city.Graph,**D_city.settings)
xs,ys,ids = D_city.get_ids()
posi = dict(zip(ids,zip(xs,ys)))

NX = True
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

bla = nx.to_directed(D_city.Graph)

for k in range(len(strecke)):
    list_strecke = list(strecke[k][:])
    list_strecke[2] = np.exp(-(np.square(list_strecke[2]))/(2*100**2))
    #strecke[k][2] = np.exp(-np.square(strecke[k][2])/2)
    bla.add_weighted_edges_from([tuple(list_strecke)])

A = D_city.sparse_adj
A2 = D_city.remove_diagsLoops(direction="directed")
A3 = nx.convert_matrix.to_scipy_sparse_matrix(bla)
#fig2,ax2 = plt.subplots(1,2)

#D_city.plot_matrix(A,figures=ax2[0])
#D_city.plot_matrix(nx.convert_matrix.to_scipy_sparse_matrix(bla),figures=ax2[1])

#plt.show()

import pygsp as ps 
G = ps.graphs.Graph(A)
G.set_coordinates([[v,z] for v,z in zip(xs,ys)])
G2 = ps.graphs.Graph(A3)
G2.set_coordinates([[v,z] for v,z in zip(xs,ys)])

fig1,ax1 = plt.subplots(2,3,figsize=(12,5))
plt.set_cmap('seismic')
plt.tight_layout()
#_ = ax1[0].spy(G.W,markersize=2)
G.compute_laplacian('combinatorial')
G.compute_fourier_basis()
G.compute_differential_operator()
G2.compute_laplacian('combinatorial')
G2.compute_fourier_basis()
G2.compute_differential_operator()
rs = np.random.RandomState(42)
sources = 20#(rs.rand(G2.n_vertices) > 0.9).astype(bool)
signal = np.zeros(G2.n_vertices)
signal[sources] = 20

_,_,we = G2.get_edge_list()
times = [0, 5, 20]
for i, t in enumerate(times):
    g = ps.filters.Heat(G2,scale=t,normalize=False)
    title = r'$\hat{{f}}({0}) = g_{{1,{0}}} \odot \hat{{f}}(0)$'.format(t)
    g.plot(alpha=1,ax=ax1[0,i],title=title)
    ax1[0,i].set_xlabel(r'$\lambda$')
    if i > 0:
        ax1[0,i].set_ylabel('')
    y = g.filter(signal)
    line, = ax1[0,i].plot(G2.e,G2.gft(y))
    labels = [r'$\hat{{f}}({})$'.format(t), r'$g_{{1,{}}}$'.format(t)]
    ax1[0, i].legend([line, ax1[0, i].lines[-3]], labels, loc='lower right')
    G2.plot(y, edges=True,edge_width=we, highlight=sources, ax=ax1[1, i], title=r'$f({})$'.format(t))
    print(np.sum(y))
    ax1[1,i].set_aspect('equal','datalim')
    ax1[1,i].margins(x=-0.3,y=-0.49)
    ax1[1,i].set_axis_off()




plt.show()
