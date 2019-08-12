import osmnx as ox 
import networkx as nx 
import numpy as np 
from matplotlib import gridspec 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm  
import matplotlib.colors as colors 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import pygsp as ps 
import pandas as pd 
import geopandas as gpd 
from descartes import PolygonPatch 
from shapely.geometry import Point, LineString, Polygon
import scipy as sc 
import collections
from pathlib import Path 

class DarmstadtNetwork:
    adj = None
    geo = dict(north=49.8815,south=49.8463,west=8.6135,east=8.6895)
    locationFile = "."
    nameFile = "d"
    Graph = nx.MultiGraph
    sparse_adj = None
    cityMap = None
    figCityMap = None
    spy = None 
    settings = dict(bgcolor="white",equal_aspect=False,node_size=10,node_color="#ff0000",node_edgecolor="none",node_zorder=2,axis_off=False,edge_color="#555555",edge_linewidth=1.5,edge_alpha=1,show=False,close=False,save=False)


    def __init__(self, geopostion=None, name=None):
        if geopostion:
            self.geo = geopostion
        if name:
            self.nameFile = name 
        else:
            try:
                self.Graph = self.load_darmstadt(self.nameFile)
                self.sparse_adj = self.extract_adjencecacy()
                self.figCityMap, self.cityMap = ox.plot_graph(self.Graph,**self.settings)
            except FileNotFoundError:
                print("Could not find file ... redownload")
                self.Graph = self.download_darmstadt(self.geo, self.nameFile, self.locationFile, show=False)
                self.sparse_adj = self.extract_adjencecacy()
                self.figCityMap, self.cityMap = ox.plot_graph(self.Graph,**self.settings)
                

    def load_darmstadt(self,name=None,show=False):
        """
        This function loads a given graphml file and plots\n
        @param:\n
            name {string}: Name of the file\n
        @optional param:\n
            show {bool} -- flag to plot  (default: {False})\n
        @return:\n
            osmnx multigraph network: File to be loaded 
        """
        if name is None:
            name = self.nameFile
        if Path("./{}.graphml".format(name)).is_file():
            print("Load Darmstadt from Graphml File ...")
            G = ox.load_graphml("{}.graphml".format(name),folder=".")
            print("Succesfully loaded")
            if show:
                ox.plot_graph(G,show=show,close=False)
            return G
        else:
            print("Graphml File doesn't exist!")
            raise FileNotFoundError("Graphml File {} does not exist".format(name + ".graphml"))

    def download_darmstadt(self, geo, name, loc, show=True):
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
        return G
    
    
    def print_stats(self, graph=None, type="basic", log=False):
        """
        This function print out the statistics of the graph
        
        @param:\n
            graph {networkx Graph}: Graph for which statistics should be calculated\n
        \n
        @param:\n
            type {str}:  setting which defines the level of output (default: {"basic"})\n
            log {bool}:  if set true print out the statistics (default: {False})\n
        \n
        @return:\n
            stats {str}: statistics of graph 
        """
        if graph is None:
            graph = self.Graph

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

    def extract_adjencecacy(self, graph=None, structure="sparse", direction="undirected"):
        """
        This function returns the adjancecy matrix of a graph\n
        
        @param:\n
            graph {graph}:  graph containing the nodes and egdes either in numpy, scipy or networkx format\n
        
        @param:\n
            structure {str}:  flag defining the kind of output graph (default: {"sparse"})\n
            direction {str}:  flag defining type of graph (default: {"undirected"})\n
        \n
        @return:\n
            [numpy/scipy graph]: type of output
        """
        if graph is None:
            graph = self.Graph

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


    def plot_matrix(self, matrix=None, figures=111):
        """
        This function plots a spy view of a graph/matrix\n
        \n 
        @param:\n
            matrix {numpy/scipy matrix} -- input matrix in 2D, either numpy or scipy format\n
        \n
        @param:\n
            figures {int}:   number of plots in a figure (default: {111})\n
            sparse {bool}:   type of input matrix (default: {True})\n
        \n
        @return:\n
            [axes]:     axes of plot
        """
        if matrix is None:
            matrix = self.sparse_adj
        if type(matrix)==sc.sparse.csr.csr_matrix:
            rows = matrix.tocoo().row
            cols = matrix.tocoo().col
            data = matrix.tocoo().data
            ax = plt.subplot(figures)
            if (matrix.shape[1]) < 50:
                im = ax.scatter(rows, cols, c=data, s=15, cmap='coolwarm')
            elif (matrix.shape[1]) < 100:
                im = ax.scatter(rows, cols, c=data, s=10, cmap='coolwarm')
            else:
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


    def plot_parallel_edges(self, ax=None, xs=None, ys=None, graph=None, color="blue"):
        """
        This function plots position of nodes where self-loops or multiedges are located\n
        @param:\n
            ax {axes}:     axes of base plot \n
            xs {[type]}:   list of x-postions of complete graph\n
            ys {[type]}:   list of y-postions of complete graph\n
            graph {numpy matrix}:  numpy array containing adjancecy matrix\n
        @param:\n
            color {str}:   setting for node color of self-loops and multiedges(default: {"blue"})
        """
        if all(v is None for v in [graph, ax, xs, ys]):
            graph = self.sparse_adj
            ax = self.cityMap
            
        if sc.sparse.issparse(graph):
            print("Sparse Matrix --- Transformiere")
            temp = self.extract_adjencecacy(self.Graph,structure="dense")
            xxs, yys = np.where(temp == np.amax(temp))
        else:
            print("Dichte Matrix --- berechne max einträge")
            xxs, yys = np.where(graph==np.amax(graph))
        xs,ys,ids = self.get_ids()
        for k in range(0,len(xxs)):
            ax.scatter(xs[xxs[k]], ys[xxs[k]],c=color)
            ax.scatter(xs[yys[k]], ys[yys[k]], c=color)
        plt.show()

    def remove_diagsLoops(self,xs=None,ys=None,graph=None,direction="undirected"):
        if all(v is None for v in [graph,xs,ys]):
            graph = self.sparse_adj
        if sc.sparse.issparse(graph) and direction=="undirected":
            print("Sparse Matrix --- Transformiere")
            temp = self.extract_adjencecacy(self.Graph, structure="dense")
            xxs, yys = np.where(temp == np.amax(temp))
        elif direction=="directed":
            # TODO check for bugs
            temp = self.extract_adjencecacy(structure="dense", direction="directed")
            xxs, yys = np.where(temp == np.amax(temp))
        else:
            print("Dichte Matrix --- berechne maximale Einträge")
            xxs, yys = np.where(graph == np.amax(graph))
        #xs, ys, ids = self.get_ids()
        for k in range(0,len(xxs)):
            if xxs[k] == yys[k]:
                temp[xxs[k],xxs[k]] = 0
            else:
                temp[xxs[k],yys[k]] = 1
                temp[yys[k],xxs[k]] = 1
        return sc.sparse.csr_matrix(temp)

    def get_laplacian(self, graph=None, kind="laplacian",show=False):
        """
        This function plots the Laplacian of a matrix\n
        @param:\n
            graph {networkx undirected}:  A undirected Graph is mandatory 
        @param:\n
            show {bool}:  flag for showing plot (default: {False})\n
        @return:\n
            [ax]:   axes of plot\n
            [L]:    Laplacian of graph 
        """
        if graph is None:
            graph = self.Graph
        # TODO refactor if only laplacians should be plotted
        if kind=="laplacian":
            if type(graph) == nx.MultiDiGraph:
                graph = nx.to_undirected(graph)
            L = nx.linalg.laplacian_matrix(graph)
            ax = self.plot_matrix(matrix=L)
            if show:
                plt.show()
            else:
                return L, ax

    def get_diag(self, graph=None, norm=False, show=False, kind="matrix"):
        """
        This function returns the diagonal elements of a matrix of a graph containing the degree\n
        @param:\n
            graph {networx}:  undirected multidigraph\n
        @param:\n
            norm {bool}:  flag if set true normalizes the diagonal elements (default: {False})\n
            show {bool}:  flag if set true plots the diagonal elements (default: {False})\n
        @return:\n
            [scipy matrix]:  scipy sparse matrix containing the diagonal elements with degree of each node
        """
        if graph is None:
            graph = self.Graph

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
            if kind=="matrix":
                ax = self.plot_matrix(matrix=D)
                plt.show()
            elif kind=="2D":
                if norm:
                    plt.plot(diagsSQ[0,:].T)
                    plt.show()
                else:
                    plt.plot(diags[0,:].T)
                    plt.show()
        if norm is None and show is None:
            return D


    def spot_diags(self,adjancecy=None, show=True, ax=None):
        if adjancecy is None:
            adjancecy = self.sparse_adj

        if type(adjancecy)==sc.sparse.csr.csr_matrix:
            locations = np.where(adjancecy.diagonal() != 0)
            if show:
                if ax:
                    ax.scatter(locations,locations,c="purple",s=30,marker='v')
                    plt.show()
                else:
                    ax_diags = self.plot_matrix(matrix=adjancecy)
                    ax_diags.scatter(locations,locations,c="purple",s=30,marker='v')
                    plt.show()
            else:
                if ax:
                    ax.scatter(locations, locations, c="purple", s=30, marker='v')
                    return ax 
                else:
                    ax_diags = self.plot_matrix(matrix=adjancecy)
                    ax_diags.scatter(locations,locations,c="purple",s=30,marker='v')
                    return ax_diags
        else:
            raise ValueError("Input matrix adjancecy must be sparse")
            

    def get_ids(self,graph=None):
        """
        This function return list containing the X- and Y-Postions and the OSMIDs of all nodes\n
        @param:\n
            graph {networks graph}:  any networkx graph coming from osmnx\n
        @return:\n
            [nodeXs]:  list containing all x positions in osmnx graph\n
            [nodeYs]:  list containing all y positions in osmnx graph\n
            [nodeID]:  list containing all IDs in osmnx graph
        """
        if graph is None:
            graph = self.Graph
        nodeXs = [x for _, x in graph.nodes(data='x')]
        nodeYs = [y for _, y in graph.nodes(data='y')]
        nodeID = [ID for ID, _ in graph.nodes(data='osmid')]
        return nodeXs, nodeYs, nodeID

    def show_citymap(self):
        fig = plt.figure()
        new = fig.canvas.manager
        new.canvas.figure = self.figCityMap
        new.canvas.ax = self.cityMap
        plt.show()


#geo = dict(north=49.8815,south=49.8463,west=8.6135,east=8.6895)
#name = "d"
#download_darmstadt(geo,name, ".", show=False)

#dtown = DarmstadtNetwork()

"""


G = dtown.Graph #load_darmstadt(name)
dtown.get_laplacian(show=True)
print(dtown.nameFile)
dtown.nameFile = "Farid"
print(dtown.nameFile)
#plt.plot(figure=dtown.figCityMap,axis=dtown.cityMap)
#figu = dtown.figCityMap
fige = plt.figure()
newMan = fige.canvas.manager
newMan.canvas.figure = dtown.figCityMap
dtown.figCityMap.set_canvas(newMan.canvas)

plt.show()

dtown.figCityMap.show()



#dtown.figCityMap.show()
#plt.show()
"""

"""
xs, ys, ids = get_ids(G)

#fig1, ax1 = ox.plot_graph(G,bgcolor="white",equal_aspect=False,node_size=10,node_color="#ff0000",node_edgecolor="none",node_zorder=2,axis_off=False,edge_color="#555555",edge_linewidth=1.5,edge_alpha=1,show=False,close=False,save=False)

A = extract_adjencecacy(G,structure="sparse")
#ax2 = plot_matrix(A,sparse=True)

ax3 = spot_diags(A,show=False)
#D = get_diag(G,show=True,kind="2D")

#plot_parallel_edges(ax1,xs, ys, A)

#A = extract_adjencecacy(G,structure="dense")
#AA = nx.to_undirected(A)
plt.show()
#get_laplacian(G,show=True)

# delete when works ax = plot_matrix(G,sparse=False)
#plt.show()
#print_stats(G,type="expert",log=True)

"""
