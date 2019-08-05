import osmnx as ox 
import networkx as nx 
import numpy as np 
from matplotlib import gridspec 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm  
import matplotlib.colors as colors 
import pygsp as ps 
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



G = load_darmstadt(name)

fig1, ax1 = ox.plot_graph(G,bgcolor="white",equal_aspect=True,node_size=10,node_color="#ff0000",node_edgecolor="none",node_zorder=2,axis_off=False,edge_color="#555555",edge_linewidth=1.5,edge_alpha=1,show=True,close=False,save=False)

#print_stats(G,type="expert",log=True)
