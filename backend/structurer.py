import functions as me
import networkx as nx

dataSources = me.data_sources("C:/Users/Anass ELBAZ/Desktop/data lake/python/backend/DL/")
graph = nx.Graph()

for source in dataSources:
    keywords = me.read_source("C:/Users/Anass ELBAZ/Desktop/data lake/python/backend/DL/" + source)
    graph = me.generate_initial_sub_graphe(graph, keywords, source)

me.add_semantic_similarity(graph)
me.add_lexical_similarity(graph)
me.clean_graph(graph)

# me.complexKnowledgePattern(graph, "short", "function")

me.draw_graph_hv(graph)
me.draw_graph_nx(graph)

