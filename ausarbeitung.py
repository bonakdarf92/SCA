from DarmstadtNetwork import DarmstadtNetwork
import networkx as nx 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from Plots.initialPlots import settup_pgf
# import osmnx as ox
# import numpy as np
# import cvxpy as cp 
# from tqdm import tqdm
# import gurobipy
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

D_city.plot_map()
D_city.show_citymap()
