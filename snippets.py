import numpy as np
import networkx as nx 
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt 

G_hierar = nx.DiGraph()
G_hierar.add_node("Root")

for i in range(3):
    G_hierar.add_node("Kind {}".format(i))
    G_hierar.add_node("Enkel {}".format(i))
    G_hierar.add_node("Enkel {}".format(i+3))
    G_hierar.add_edge("Kind {}".format(i),"Root")
    G_hierar.add_edge("Enkel {}".format(i), "Kind {}".format(i))
    G_hierar.add_edge("Enkel {}".format(i+3), "Kind {}".format(i))
G_hierar = G_hierar.to_directed()
plt.title("Hierarchie Graph")
pos = graphviz_layout(G_hierar, prog='dot')
nx.draw(G_hierar, pos, with_labels=False, arrows=True,node_color="blue")
nx.draw_networkx_nodes(G_hierar,pos,nodelist=["Root"], node_color="blue")
plt.show()

G_ring = nx.DiGraph()
G_ring.add_node("k0")
for i in range(1,9):
    G_ring.add_node("k{}".format(i))
    G_ring.add_edge("k{}".format(i-1), "k{}".format(i))
    G_ring.add_edge("k{}".format(i), "k{}".format(i-1))
G_ring.add_edge("k8", "k0")

plt.title("Ring Graph")
pos = nx.circular_layout(G_ring)
nx.draw(G_ring,pos,with_labels=False,arrows=True,node_color="blue")
plt.show()

plt.title("Vollständiger Graph")
G_voll = nx.complete_graph(9)
G_voll = G_voll.to_directed()
pos = nx.circular_layout(G_voll)
nx.draw(G_voll,pos,arrows=True,node_color="blue")
plt.show()
fig,ax1 = plt.subplots(1,2)

G_stern_zent = nx.star_graph(9)
pos = nx.spring_layout(G_stern_zent)
G_stern_zent = G_stern_zent.to_directed()
nx.draw(G_stern_zent,pos,ax=ax1[0],node_color="red",edgelist=[(k,0) for k in range(1,10)])
nx.draw_networkx_nodes(G_stern_zent,pos, nodelist=[0],node_color="blue",ax=ax1[0])
ax1[0].set_title("Zentraler Graph")

G_stern_dezent = nx.star_graph(9)
G_stern_dezent = G_stern_dezent.to_directed()
nx.draw(G_stern_dezent,pos, arrows=True,ax=ax1[1],node_color="red")
nx.draw_networkx_nodes(G_stern_dezent,pos,nodelist=[k for k in range(1,10)],node_color="blue",ax=ax1[1])
ax1[1].set_title("Dezentraler Graph")
plt.show()


plt.title("Zufälliger Graph")
G_rand = nx.random_lobster(8,0.66,0.1,seed=42)
pos = nx.spring_layout(G_rand)
G_rand = G_rand.to_directed()
nx.draw(G_rand,pos,arrows=True,node_color="blue")
plt.show()


plt.title("Zufälliger Graph")
nx.draw(G_rand,pos,arrows=True,node_color="blue",with_labels=False)
labels = {}
labels[0] = r'$\alpha$'
labels[1] = r"$\beta$"
labels[2] = r"$\gamma$"
labels[11] = r"$\delta$"
nx.draw_networkx_labels(G_rand,pos,labels,font_color="red",font_size=16)
plt.show()

