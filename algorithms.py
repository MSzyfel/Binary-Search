import heapq

import numpy as np
import networkx as nx
from tree import Tree
from decision_tree import DecisionTree
from max_heap_object import MaxHeapObj


def dp_tree(tree: Tree, dp=None):
    if dp is None:
        dp = {}
    if len(tree) <= 2:
        nodes = tree.nodes
        v = min(nodes, key=lambda v: tree.weight(v))
        return DecisionTree(v)
    candidate_dts = []
    for v in tree.nodes():
        tree_copy = tree.copy()
        dt = DecisionTree(v)
        for cc in tree_copy.ccs(v):
            cc_hash = hash(cc)
            if dp[cc_hash] is not None:
                dt_cc = dp[cc_hash]
            else:
                dt_cc = dp_tree(cc, dp)
            dt.attach_subtree(dt_cc, v)
        candidate_dts.append(dt)

    dc = min(candidate_dts, key=lambda dt: dt.cost())
    dp[hash(tree)] = dc
    return dc


def tree_search_cicalese_inspired(tree: Tree, t: int):
    if len(tree) <= t:
        return dp_tree(tree)
    else:
        centroids = set()
        subtrees = [MaxHeapObj(tree.copy())]
        for _ in range(0, t):
            subtree = subtrees[len(subtrees)-1].val
            centroid = subtree.centroid()
            centroids.add(centroid)
            c_ccs = subtree.ccs(centroid)
            for cc in c_ccs:
                heapq.heappush(subtrees, MaxHeapObj(cc))
        subtree = subtrees[len(subtrees)-1].val
        x = subtree.vertices_of_degree_at_least(3)
        y = tree.minimal_subtree_with_contracted_paths(x)
        dt_y = dp_tree(y)
        for p in tree.minimal_subtree(x).ccs(y.nodes):
            dt_p = dp_tree(p)
            dt_y.attach_sub_dt(tree, p, dt_p)
        for h in tree.ccs(tree.minimal_subtree(x)):
            dt_h = tree_search_cicalese_inspired(h, t)
            dt_y.attach_sub_dt(tree, h, dt_h)
        return dt_y





