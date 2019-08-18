# imports
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import difflib
import numpy as np
import spacy
import os
import numpy as np
import pandas as pd
import holoviews as hv
import networkx as nx
from holoviews import opts
from bokeh.io import output_file, save, show
import hvplot.networkx as hvnx

#constants and declarations
THK = 0.7
THKMIN = 0.6
SYNTAXIC_LIMEN = 0.65
hv.extension('bokeh')


#functions
def data_sources(path):
        return os.listdir(path)

def remove_value_from_list(the_list, val):
        for _ in range(the_list.count(val)):
                the_list.remove(val)

def remove_values_from_list(the_list, vals):
    for i in range(len(vals)): 
            remove_value_from_list(the_list, vals[i])

def generate_keywords(data):
    k = nltk.word_tokenize(data)
    remove_values_from_list(k, ['.', ',', '(', ')', '[', ']', '{', '}','a', ';', '"', ':'])
    return k

def read_source(file_name):
    source = open(file_name, "r")
    data = source.read()
    source.close()
    k= generate_keywords(data)
    return k

def generate_initial_sub_graphe(G, keywords, source):
    # creation d'un graphe vide
    #G = nx.Graph()
    # ajout des neouds 
    G.add_node(source, type='complexe')
    for word in keywords:
            G.add_node(word, type='simple')
    # ajout des arcs
    for node in keywords:
        G.add_edge(source, node, weight=1, color='b')
    print("Initial sub graphe constructed : ", source)
    return G

def distance(word1, word2):
    seq = difflib.SequenceMatcher(None,word1,word2)
    return seq.ratio()

def add_lexical_similarity(G):
    kd_set = []
    for w1 in G.nodes:
            for w2 in G.nodes:
                    if w1 != w2:
                            kd = distance(w1, w2)
                            kd_set.append(kd)
    kd_max = np.amax(kd_set)
    for w1 in G.nodes:
            for w2 in G.nodes:
                    if w1 != w2:
                            kd = distance(w1, w2)
                            if kd >= THK*kd_max and kd >= THKMIN:
                                print("lexical similarity added : (", w1, ", ", w2, ", ", kd, ")")
                                G.add_edge(w1, w2, weight=5, color='r')
                                G.add_node(w1, type='complexe', relationships=G.node[w2])
                                G.add_node(w2, type='complexe', relationships=G.node[w2])

def add_semantic_similarity(G):
        keywords = G.nodes
        nlp = spacy.load('en_core_web_lg')
        for w1 in keywords:
                for w2 in keywords:
                        if w1 != w2:
                                doc1 = nlp(w1)
                                doc2 = nlp(w2)
                                sim = doc1.similarity(doc2)
                                if sim > SYNTAXIC_LIMEN:
                                        print("semantic similarity added : (", w1, ", ", w2, ", ", sim, ")")
                                        G.add_edge(w1, w2, weight=5, color='g')
                                        G.add_node(w1, type='complexe', relationships=G.node[w1])
                                        G.add_node(w2, type='complexe', relationships=G.node[w2])

def draw_graph_nx(G):
        #pos = nx.spring_layout(G)
        pos = nx.fruchterman_reingold_layout(G)
        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]
        weights = [G[u][v]['weight'] for u,v in edges]
        nx.draw(G, pos,with_labels=True, node_size=1000, node_color="skyblue", alpha=0.9, edges=edges, edge_color=colors, width=weights)
        plt.show()
        
def draw_graph_hv(G):
        pos = nx.fruchterman_reingold_layout(G)
        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]
        weights = [G[u][v]['weight'] for u,v in edges]
        options = {
                'node_size': 2000,
                'node_color': '#A0CBE2',
                'edge_width': weights,
                'edge_color': colors,
                'width': 1800,
                'height': 900,
                'with_labels': True,
                'alpha': 0.8
        }
        g = hvnx.draw(G, pos ,**options)
        renderer = hv.renderer('bokeh')
        renderer.save(g, 'graph.html')
        plot = renderer.get_plot(g).state
        output_file("graph.html")
        save(plot, 'graph.html')
        show(plot)

def complexKnowledgePattern(G,source,target):
        try:
                CKPList = nx.algorithms.shortest_paths.unweighted.bidirectional_shortest_path(G, source, target)
                for i in range(len(CKPList) - 1):
                        G.add_edge(CKPList[i], CKPList[i+1], weight=10, color='g')
                return CKPList
        except nx.NetworkXNoPath:
                print('No CKP founded :/')
                return None

def clean_graph(G):
        for node in G.nodes:
                list_neighbors = G.neighbors(node)
                if len(list_neighbors) < 2:
                        G.remove_node(node)
