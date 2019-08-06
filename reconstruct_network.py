import osmnx as ox 
import networkx as nx 
import numpy as np 
from matplotlib import gridspec 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm  
import matplotlib.colors as colors 
#import pygsp as ps 
import pandas as pd 
import geopandas as gpd 
from descartes import PolygonPatch 
from shapely.geometry import Point, LineString, Polygon
import scipy as sc 
import collections
from pathlib import Path 


# TODO write some argparse flags if needed
def download_darmstadt(geo,name,loc,show=True):
    """
    This functions downloads and save the network city map of darmstadt\n
    @param:\n
        geo {dict}: [Dictionary of latitude and longitude GPS coordinates]\n
        name {String}: [name of graphml file]\n
        loc {String}: [save location]\n
        show {bool}: [flag which plots the downloaded network city map] (default: {True})
    """
    
    if all(k in geo for k in ("north","south","west","east")):
        print("Load Darmstadt from Openstreetmap ... ")
        G = ox.core.graph_from_bbox(north=geo["north"],south=geo["south"],west=geo["west"],east=geo["east"],network_type='drive')
        print("Save Darmstadt ... ")
        ox.save_load.save_graphml(G,filename="{}.graphml".format(name),folder=loc)
        print("Saved")
    else:
        print("Geoposition is missing an entity e.g. north,south, etc. ")
    if show:    
        print("Plot Darmstadt ... ")
        ox.plot_graph(G,show=True,close=False)


geo = dict(north=49.8815,south=49.8463,west=8.6135,east=8.6895)
name = "d"
#download_darmstadt(geo,name, ".", show=False)

def load_darmstadt(name,show=False):
    """
    This function loads a given graphml file and plots\n
    @param:\n
        name {string}: Name of the file\n
    @optional param:\n
        show {bool} -- flag to plot  (default: {False})\n
    @return:\n
        osmnx multigraph network: File to be loaded 
    """
    if Path("./{}.graphml".format(name)).is_file():
        print("Load Darmstadt from Graphml File ...")
        G = ox.load_graphml("{}.graphml".format(name),folder=".")
        print("Succesfully loaded")
        if show:
            ox.plot_graph(G,show=show,close=False)
        return G
    else:
        print("Graphml File doesn't exist!")
        return -1


def print_stats(graph,type="basic",log=False):
    if type=="basic":
        stats = ox.basic_stats(graph)
        for key,value in stats.items():
            stats[key] = value
        if log:
            print(pd.Series(stats))
        else:
            return stats
    if type=="pro":
        stats = ox.extended_stats(graph)
        for key,value in stats.items():
            stats[key] = value
        if log:
            print(pd.Series(stats))
        else:
            return stats 
    if type=="expert":
        stats = ox.extended_stats(graph, connectivity=True, ecc=True, cc=True, bc=True)
        for key,value in stats.items():
            stats[key] = value
        if log:
            print(pd.Series(stats))
        else:
            return stats
    if type=="all":
        stats = ox.extended_stats(graph, connectivity=True, ecc=True, anc=True, cc=True, bc=True)
        for key,value in stats.items():
            stats[key] = value
        if log:
            print(pd.Series(stats))
        else:
            return stats 


def extract_adjencecacy(graph,structure="sparse",direction="undirected"):
    if direction=="undirected":
        graph_un = nx.to_undirected(graph)
        if structure=="sparse":
            return nx.convert_matrix.to_scipy_sparse_matrix(graph_un)
        else:
            return nx.convert_matrix.to_numpy_matrix(graph_un)
    else:
        if structure=="sparse":
            return nx.convert_matrix.to_scipy_sparse_matrix(graph)
        else:
            return nx.convert_matrix.to_numpy_matrix(graph)


def plot_matrix(matrix, figures=111, sparse=True):
    if sparse:
        rows = matrix.tocoo().row
        cols = matrix.tocoo().col
        data = matrix.tocoo().data
        ax = plt.subplot(figures)
        im = ax.scatter(rows, cols, c=data, s=1, cmap='coolwarm')
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.invert_yaxis()
        return ax
    else:
        ax = plt.subplot(figures)
        matrix = nx.to_numpy_matrix(matrix)
        x,y = np.argwhere(matrix==1).T
        im = ax.scatter(x,y)
        #im = ax.scatter(x,y, s=1, cmap='coolwarm')
        #cbar = ax.figure.colorbar(im, ax=ax)
        ax.invert_yaxis()
        return ax


def plot_laplacian(graph, kind="laplacian",show=False):
    if kind=="laplacian":
        if type(graph) == nx.MultiDiGraph:
            graph = nx.to_undirected(graph)
        L = nx.linalg.laplacian_matrix(graph)
        ax = plot_matrix(L)
        #rows = L.tocoo().row
        #cols = L.tocoo().col 
        #data = L.tocoo().data
        #ax = plt.subplot(111)
        #im = ax.scatter(rows, cols, c=data, s=1, cmap='coolwarm')
        #cbar = ax.figure.colorbar(im,ax=ax)
        #ax.invert_yaxis()
        if show:
            plt.show()
        else:
            return ax

# TODO implement show part
def get_diag(graph,norm=False, show=False):
    if type(graph)==nx.MultiDiGraph:
        graph = nx.to_undirected(graph)
    A = nx.to_scipy_sparse_matrix(graph, nodelist=list(graph), weight='weight', format='csr')
    n,m = A.shape
    diags = A.sum(axis=1).flatten()
    D = sc.sparse.spdiags(diags, [0], m, n, format='csr')
    if norm:
        diagsSQ = 1.0/sc.sqrt(diags)
        D = sc.sparse.spdiags(diagsSQ, [0], m, n, format='csr')
    if show:
        pass


def get_ids(graph):
    nodeXs = [x for _, x in graph.nodes(data='x')]
    nodeYs = [y for _, y in graph.nodes(data='y')]
    nodeID = [id for id, _ in graph.nodes(data='osmid')]
    return nodeXs, nodeYs, nodeID


def plot_parallel_edges(ax, xs, ys, graph, color="blue"):
    if sc.sparse.issparse(graph):
        print("Sparse Matrix --- Transformiere")
        temp = extract_adjencecacy(graph,structure="dense")
        xxs, yys = np.where(temp == np.amax(temp))
    else:
        print("Dichte Matrix --- berechne max einträge")
        xxs, yys = np.where(graph==np.amax(graph))
    for k in range(0,len(xxs)):
        ax.scatter(xs[xxs[k]], ys[xxs[k]],c=color)
        ax.scatter(xs[yys[k]], ys[yys[k]], c=color)
    plt.show()

G = load_darmstadt(name)
xs, ys, ids = get_ids(G)

fig1, ax1 = ox.plot_graph(G,bgcolor="white",equal_aspect=False,node_size=10,node_color="#ff0000",node_edgecolor="none",node_zorder=2,axis_off=False,edge_color="#555555",edge_linewidth=1.5,edge_alpha=1,show=False,close=False,save=False)

A = extract_adjencecacy(G,structure="dense")

get_diag(G)

# TODO comment function
plot_parallel_edges(ax1,xs, ys, A)

#A = extract_adjencecacy(G,structure="dense")
#AA = nx.to_undirected(A)
#plt.show()
plot_laplacian(G,show=True)

# delete when works ax = plot_matrix(G,sparse=False)
plt.show()
#print_stats(G,type="expert",log=True)


