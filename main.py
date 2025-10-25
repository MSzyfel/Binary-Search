from math import log2, sqrt

import networkx as nx

from tree import Tree
from algorithms import tree_search_cicalese_inspired, ranking_based_dt, qptas_dereniowski_inspired, dp_tree, \
    k_up_modularity_algorithm, average_case_trees_fptas, centroid_dt


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


# tree = Tree.make_k_up_modular(n=400, k=5)

def uniform_costs():
    return 1


# for n in [10, 25, 50, 100, 200]:
#     tree = Tree(n=n)
#     tree.to_pdf(filename=f"outputs/tree_{n}.pdf", orientation='horizontal', attribute='c', type=float,
#                 proportions=(6, 8))
#     # dt = average_case_trees_fptas(tree=tree)
#     # print(dt.cost(crit='average'))
#     # print(dt.cost(crit='worst'))
#     # dt.draw(attribute='c', type=float, orientation='horizontal')
#     t = int(2 ** (sqrt(log2(len(tree)))))
#     dt1 = tree_search_cicalese_inspired(tree, t, qptas_dereniowski_inspired)
#     cost = dt1.cost()
#     print(cost)
#     dt1.to_pdf(filename=f"outputs/dt_{n}.pdf", attribute='c', type=float, proportions=(6, 8))
#     with open("outputs/costs.txt", "a", encoding="utf-8") as f:
#         f.write(f"n = {n}, cost = {cost}\n")

T1 = Tree.probabilistic(20)
T2 = Tree.recursive(20)
T3 = Tree.preferential(20)
T1.draw(attribute='c', type=float)
T2.draw(attribute='c', type=float)
T3.draw(attribute='c', type=float)
