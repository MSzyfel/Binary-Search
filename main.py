from math import log2

import networkx as nx

from tree import Tree
from algorithms import tree_search_cicalese_inspired

tree = Tree(n=50)
print(tree.seed)
tree.draw()
t = int(2 ** (log2(log2(len(tree)))))
dt = tree_search_cicalese_inspired(tree, t)
dt.draw('horizontal')
print(dt.cost())


