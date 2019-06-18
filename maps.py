#import geopandas as gpd
#import geoviews as gv
#import cartopy.crs as ccrs 
import osmnx as ox 
import matplotlib.pyplot as plt
import pandas as pd 
import networkx as nx
import numpy as np


openDarmstadt = pd.read_csv("darmstadt.csv")
x = openDarmstadt['Latitude']
y = openDarmstadt['Longitude']
#G = ox.graph_from_place('Darmstadt, Hessen', network_type='drive')
#G = ox.footprints.create_footprints_gdf(west=8.6295,south=49.8663,east=8.6695,north=49.8815, footprint_type='building')
#street = ox.core.graph_from_bbox(north=49.8815,south=49.8663,west=8.6295,east=8.6695,network_type='drive')
#darmstadt = ox.core.graph_from_place('Darmstadt,Germany',network_type='drive')
#darm = ox.core.graph_from_point((49.8741,8.6510),distance=6000, network_type='drive', simplify=True)
darm2 = ox.core.graph_from_file('darm_osm')
stats = ox.stats.extended_stats(darm2)
D_mat = nx.convert_matrix.to_numpy_matrix(darm2)
#print(type(darm2.get_node()))
print(stats)
print(D_mat)
#names = darm.edges.data('name') 
#print(names)
#ox.save_load.save_graph_osm(darm,filename='darm_osm',folder='.')
#ox.save_load.save_graph_shapefile(darm,filename='darm_shape',folder='.')
#ox.save_load.save_graphml(darm,filename='darm.graphml',folder='.',gephi=True)
#ox.footprints.plot_footprints(G)
#print(darm.nodes)
fig, ax = ox.plot.plot_graph(darm2, axis_off=False, show=False, close=False, save=False,fig_height=10,fig_width=15)

#fig, ax = ox.plot.plot_graph(darm2, axis_off=False, show=False, close=False, save=False)
ax.scatter(y,x,marker="*",color="red",label='hauptstraßen')
#ax.plot(x[690],y[690],marker="*",color="red")
plt.legend()
#mng = plt.get_current_fig_manager()
#mng.window.showMaximized()
plt.savefig('darmstadtNetzwerk')
#G = ox.graph_from_point((37.79, -122.41), distance=750, network_type='all')
#ox.plot_graph(G)

#gv.extension('bokeh')
print("ok")
#tiles = gv.tile_sources.Wikipedia
#test = gpd.dataset.available()