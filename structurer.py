'''
Algorithme:
- lecture de la source
- spliter les mots
- creer un graphe
- traitement par BabelNet
- traitement par N-Gram
'''

import nltk
import networkx as nx
import matplotlib.pyplot as plt

def remove_values_from_list(the_list, val):
    for i in range(the_list.count(val)):
        the_list.remove(val)

source = open("source.txt", "r")
data = source.read()
source.close()

keywords = nltk.word_tokenize(data)


remove_values_from_list(keywords, '.')
remove_values_from_list(keywords, ',')

#print(keywords)

G = nx.Graph() # Right now G is empty
# Add a node
#G.add_node(1) 
#G.add_nodes_from([1,2,3,4]) # You can also add a list of nodes by passing a list argument

src_node = 'source'
G.add_node(src_node, type='complexe')
G.add_nodes_from(keywords, type='simple')


for node in G.nodes:
    G.add_edge(src_node, node)

# Add edges 
#G.add_edge(1,2)
#e = (2,3)
#G.add_edge(*e) # * unpacks the tuple
#G.add_edges_from([(1,2), (1,3), (1,4), (2,3)]) # Just like nodes we can add edges from a list
#nx.draw(G)
#nx.draw_networkx(G, with_labels=True)
nx.draw(G, with_labels=True, node_size=3000, node_color="skyblue", alpha=0.9, pos=nx.fruchterman_reingold_layout(G))
# show graph
plt.title("KeyWords")
plt.show()

