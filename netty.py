import networkx as nx 
my_graph = nx.Graph()
edges = nx.read_edgelist('roadNet-CA.txt')
# nodes list missing
my_graph.add_edges_from(edges.edges())
nx.draw(my_graph, with_labels=False)
