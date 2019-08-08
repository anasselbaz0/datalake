# imports
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import difflib
import numpy as np
import spacy

#constants
THK = 0.7
THKMIN = 0.5
SYNTAXIC_LIMEN = 0.6

#functions
def remove_value_from_list(the_list, val):
    for i in range(the_list.count(val)):
            the_list.remove(val)

def remove_values_from_list(the_list, vals):
    for i in range(len(vals)): 
            remove_value_from_list(the_list, vals[i])

def generate_keywords(data):
    k = nltk.word_tokenize(data)
    remove_values_from_list(k, ['.', ',', '(', ')', '[', ']', '{', '}','a', ';', '"'])
    return k

def read_source(file_name):
    source = open(file_name, "r")
    data = source.read()
    source.close()
    k= generate_keywords(data)
    print(k)
    return k

def generate_initial_graphe(keywords):
    # creation d'un graphe vide
    G = nx.Graph()
    # ajout des neouds 
    G.add_node('source', type='complexe')
    for word in keywords:
            G.add_node(word, type='simple')
    # ajout des arcs
    for node in G.nodes:
        G.add_edge('source', node, weight=1, color='b')
    print("Initial graphe constructed")
    return G

def distance(word1, word2):
    seq = difflib.SequenceMatcher(None,word1,word2)
    return seq.ratio()

def add_lexical_similarity(G):
    kd_set = []
    keywords = G.nodes
    for w1 in keywords:
            for w2 in keywords:
                    if w1 != w2:
                            kd = distance(w1, w2)
                            kd_set.append(kd)
    kd_max = np.amax(kd_set)
    for w1 in keywords:
            for w2 in keywords:
                    if w1 != w2:
                            kd = distance(w1, w2)
                            if kd >= THK*kd_max and kd >= THKMIN:
                                print("lexical similarity added : (", w1, ", ", w2, ", ", kd, ")")
                                G.add_edge(w1, w2, weight=3, color='r')

def draw_graph(G):
        pos = nx.spring_layout(G)
        #pos = nx.fruchterman_reingold_layout(G)
        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]
        weights = [G[u][v]['weight'] for u,v in edges]
        nx.draw(G, pos,with_labels=True, node_size=1000, node_color="skyblue", alpha=0.9, edges=edges, edge_color=colors, width=weights)
        plt.show()

def add_semantic_similarity(G):
        keywords = G.nodes
        nlp = spacy.load('en_core_web_md')
        for w1 in keywords:
                for w2 in keywords:
                        if w1 != w2:
                                doc1 = nlp(w1)
                                doc2 = nlp(w2)
                                sim = doc1.similarity(doc2)
                                if sim > SYNTAXIC_LIMEN:
                                        print("semantic similarity added : (", w1, ", ", w2, ", ", sim, ")")
                                        G.add_edge(w1, w2, weight=3, color='g')