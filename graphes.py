import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph() # Right now G is empty
# Add a node
#G.add_node(1) 
G.add_nodes_from([1,2,3,4]) # You can also add a list of nodes by passing a list argument
# Add edges 
#G.add_edge(1,2)
#e = (2,3)
#G.add_edge(*e) # * unpacks the tuple
G.add_edges_from([(1,2), (1,3), (1,4), (2,3)]) # Just like nodes we can add edges from a list
nx.draw(G)
# show graph
plt.show()