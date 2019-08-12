from SCA.DarmstadtNetwork import DarmstadtNetwork
from SCA.Plots import initialPlots
import networkx as nx 
import pygsp as ps 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
from matplotlib import colorbar, cm



# # TODO works fine
# rs = np.random.RandomState(42)
# W = rs.uniform(size=(30,30))
# W[W < 0.93] = 0
# W = W + W.T 
# np.fill_diagonal(W,0)
# G = ps.graphs.Graph(W)
# G.set_coordinates()
# G.plot()
# plt.show()

# print(G.is_connected())
# print(type(G.L))
# L_new = G.L.tocsr()
# print(type(L_new))
dtown = DarmstadtNetwork()
# ax = dtown.plot_matrix(matrix=L_new)
# plt.show()

# # TODO works fine
# ad_ber = dtown.remove_diagsLoops()
# ax3 = dtown.plot_matrix(matrix=ad_ber)
# plt.title("Bereinigt")
# plt.show()
# graphg = ps.graphs.Graph(ad_ber)
xs, ys, _ = dtown.get_ids()
# graphg.set_coordinates([[v,z] for v,z in zip(xs,ys)])
# print(graphg.plotting['vertex_color'])
# graphg.plotting['vertex_color'] = (0.8, 0.15, 0.03, 1)
# print(graphg.plotting['vertex_color'])
# indices = [[v,z] for v,z in zip(ad_ber.tocoo().row,ad_ber.tocoo().col)]
# reihen = np.random.choice(ad_ber.tocoo().row, int(graphg.N*0.1),replace=False)
# spalten = np.random.choice(ad_ber.tocoo().col,int(graphg.N*0.1),replace=False)
# ad_orig = dtown.sparse_adj
# ind_sel = np.random.choice(range(0,len(indices)),int(graphg.N*0.1),replace=False)
# sel = [indices[k] for k in ind_sel]
# weights = np.random.randint(2,30,len(sel))
# k,j = [k for k,_ in sel],[j for _,j in sel]
# ax5 = graphg.plot(vertex_size=30)
# plt.show()


A1 = dtown.sparse_adj       # sparse matrix of dtown with self_loops
#A2 = dtown.extract_adjencecacy(direction="directed")
A2 = dtown.remove_diagsLoops(direction="directed")      # sparse matrix of dtown directed
D_g = ps.graphs.Graph(A2)                               # pygsp graph of dtown
D_g.set_coordinates([[v,z] for v,z in zip(xs,ys)])      # set x,y coordinates
D_g.compute_laplacian('combinatorial')
D_g.compute_fourier_basis()                             # computation of fourierbasis
D_g.compute_differential_operator()                     # computation of Delta_g
#x = np.random.randint(20,50,size=D_g.N)                 # ground truth data on edges
x = np.copysign(np.ones(D_g.N),D_g.U[:,33])
#eig_c = plt.cm.coolwarm((np.clip(x,2,10)-2)/8.).squeeze()
#e = (eig_c[0][0], eig_c[0][1], eig_c[0][2],eig_c[0][3])
#D_g.plotting['vertex_color'] = eig_c
fig,axes = plt.subplots(2,2,figsize=(12,4))
D_g.plot(x,vertex_size=40,ax=axes[0,0])
axes[0,0].set_title('Vollständiges Signal')
#plt.show()
rs = np.random.RandomState(42)
x_loss = (rs.rand(D_g.N) > 0.6 ).astype(float)
x_subsample = x_loss * (x)# + 0.1 * rs.standard_normal(D_g.N))
#D_g.plot(np.copysign(np.ones(D_g.N), D_g.U[:,5]),vertex_size=30)
D_g.plot(x_subsample,vertex_size=40,ax=axes[0,1])
axes[0,1].set_title('Fehlendes Signal')
#fig.tight_layout()
#plt.set_cmap('coolwarm')
#plt.show()

import pyunlocbox as pl 
gamma = 0.1 
d = pl.functions.dummy()
r = pl.functions.norm_l1()
f = pl.functions.norm_l2(w=x_loss,y=x_subsample,lambda_=gamma)
L = D_g.D.T.toarray()

step = 0.999 / (1+np.linalg.norm(L))

solver = pl.solvers.mlfbf(L=L, step=step)
x0 = x_subsample.copy()
prob1 = pl.solvers.solve([d,r,f],solver=solver,x0=x0,rtol=0,maxit=1000)

D_g.plot(prob1['sol'],vertex_size=40,ax=axes[1,0])
axes[1,0].set_title('Rekonstruiertes Signal')
D_g.plot(x - prob1['sol'],vertex_size=40,ax=axes[1,1])
axes[1,1].set_title('Differenz Rekons. - Orig')
#fig.colorbar(cm.)
fig.tight_layout()
plt.set_cmap('coolwarm')
plt.show()

r = pl.functions.norm_l2(A=L,tight=False)
step = 0.999 / np.linalg.norm(np.dot(L.T, L) + gamma * np.diag(x_loss),2)
solver = pl.solvers.gradient_descent(step=step)
x0 = x_subsample.copy()
prob2 = pl.solvers.solve([r,f],solver=solver,x0=x0,rtol=0,maxit=1000)
fig2, axes_diff = plt.subplots(1,2,figsize=(12,4))
D_g.plot(x - prob1['sol'],vertex_size=50,ax=axes_diff[0])
axes_diff[0].set_title(r'error $\Delta_{mblf}$')
D_g.plot(x - prob2['sol'],vertex_size=50,ax=axes_diff[1])
axes_diff[1].set_title(r'error $\Delta_{gd}$')
fig2.tight_layout()
plt.set_cmap('coolwarm')
plt.show()



from SCA.stela import stela_lasso, soft_thresholding
y_stela = np.dot(x_loss,x) 
mu = 0.01 * np.linalg.norm(np.dot(y_stela,x_loss),np.inf)
objval, x_stela, error_stela = stela_lasso(x_loss,y_stela,mu,10000)
fig3,axes_diff = plt.subplots(3,1,figsize=(12,4))
axes_diff[0].errorbar(range(0,len(x)),x,yerr=x-prob1['sol'],ecolor='red')
axes_diff[1].errorbar(range(0,len(x)),x,yerr=x-prob2['sol'],ecolor='red')
axes_diff[2].errorbar(range(0,len(x)),x,yerr=x-x_stela, ecolor='green')
plt.show()


