# needs mayavi2
# run with ipython -wthread
import networkx as nx
import numpy as np
G = nx.DiGraph()
G.add_edge('x','a', capacity=0.42)
G.add_edge('x','b', capacity=0.31)
G.add_edge('x','c', capacity=0.91)
G.add_edge('x','d', capacity=0.87)
G.add_edge('x','e', capacity=0.79)
G.add_edge('a','c', capacity=1)
G.add_edge('a','b', capacity=1)
G.add_edge('b','c', capacity=1)
G.add_edge('b','d', capacity=1)
G.add_edge('d','e', capacity=1)
G.add_edge('a','y', capacity=0.58)
G.add_edge('b','y', capacity=0.79)
G.add_edge('c','y', capacity=0.13)
G.add_edge('d','y', capacity=0.09)
G.add_edge('e','y', capacity=0.21)
R = nx.algorithms.flow.boykov_kolmogorov(G, 'x', 'y')
flow_value = nx.maximum_flow_value(G, 'x', 'y')
print(flow_value)
source_tree, target_tree = R.graph['trees']
partition = (set(source_tree), set(G) - set(source_tree))