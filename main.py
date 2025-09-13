from math import log2, sqrt

import networkx as nx

from tree import Tree
from algorithms import tree_search_cicalese_inspired, ranking_based_dt, qptas_dereniowski_inspired, dp_tree, k_up_modularity_algorithm

# max_size = 500
# summaric_ratio = 0.0
# for n in range(1, max_size + 1):
#     print(f"prcessing n = {n}")
#     tree = Tree.make_k_up_modular(n=n, k=5)
#     #tree.draw(attribute='c', type = float)
#     if n < 4:
#         t = 1
#     else:
#         t = int(2 ** (sqrt(log2(len(tree)))))
#     dt1 = tree_search_cicalese_inspired(tree, t, qptas_dereniowski_inspired)
#
#     dt2 = k_up_modularity_algorithm(tree)
#     summaric_ratio += dt2.cost()/dt1.cost()
#
# print(summaric_ratio/max_size)


tree = Tree.make_k_up_modular(n=400, k=5)
tree.draw(attribute='c', type = float)

t = int(2 ** (sqrt(log2(len(tree)))))
dt1 = tree_search_cicalese_inspired(tree, t, qptas_dereniowski_inspired)

dt2 = k_up_modularity_algorithm(tree)

print(dt1.cost())
print(dt2.cost())
dt1.draw(attribute='c', type = float)
dt2.draw(attribute='c', type = float)