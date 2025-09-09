from math import log2, sqrt

import networkx as nx

from tree import Tree
from algorithms import tree_search_cicalese_inspired, ranking_based_dt, qptas_dereniowski_inspired, dp_tree, k_up_modularity_algorithm

tree = Tree(n=150)
tree.draw(attribute='c', type = float)
# t = int(2 ** (log2(log2(len(tree)))))
# dt = tree_search_cicalese_inspired(tree, t)
# dt = ranking_based_dt(tree)
# tree.draw(attribute='r')
# # dt.draw('horizontal', attribute='r')
t = int(2 ** (sqrt(log2(len(tree)))))
dt1 = tree_search_cicalese_inspired(tree, t, qptas_dereniowski_inspired)

dt2 = k_up_modularity_algorithm(tree)
print(dt1.cost())
print(dt2.cost())


