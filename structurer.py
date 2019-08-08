'''
Algorithme:
- lecture de la source
- spliter les mots
- creer un graphe
- traitement par BabelNet
- traitement de similarite lexical
'''

import functions as me

keywords = me.read_source("source.txt")
graph = me.generate_initial_graphe(keywords)
me.add_lexical_similarity(graph)
me.add_semantic_similarity(graph)
me.draw_graph(graph)
