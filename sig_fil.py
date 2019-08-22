import numpy as np 
import matplotlib.pyplot as plt 
#import pygsp_old as pg 
import pygsp as pg
from pygsp.learning import classification_tikhonov 
G = pg.graphs.Sensor(seed=42)
G.compute_fourier_basis()

g = pg.filters.Expwin(G,band_max=0.5)
fig,axes = plt.subplots(1,3,figsize=(12,4))
fig.subplots_adjust(hspace=0.5)

x = np.random.RandomState(1).normal(size=G.N)

x = 3*x/np.linalg.norm(x)

y = g.filter(x)
x_hat = G.gft(x).squeeze()
y_hat = G.gft(y).squeeze()

limits = [x.min(), x.max()]

G.plot(x, limits=limits, ax=axes[0],title='Input Signal $x$ in vertex domain')
axes[0].text(0,-0.1,'$x^T L x = {:.2f}$'.format(G.dirichlet_energy(x)))
axes[0].set_axis_off()
g.plot(ax=axes[1], alpha=1)
