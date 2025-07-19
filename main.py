from math import log2

import networkx as nx

from tree import Tree
from algorithms import tree_search_cicalese_inspired, ranking_based_dt

tree = Tree(n=10, seed = 3)
print(tree.seed)
# t = int(2 ** (log2(log2(len(tree)))))
# dt = tree_search_cicalese_inspired(tree, t)
dt = ranking_based_dt(tree)
tree.draw(attribute='r')
dt.draw('horizontal', attribute='r')
print(dt.cost())


