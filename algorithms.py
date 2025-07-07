import heapq

import numpy as np
import networkx as nx
from tree import Tree
from decision_tree import DecisionTree
from max_heap_object import MaxHeapObj


def DP(tree: Tree, dp=None):
    if dp is None:
        dp = {}
    if len(tree) <= 2:
        nodes = tree.nodes
        v = min(nodes, key=lambda v: v.get('w'))
    candidate_dts = []
    for v in tree.nodes():
        tree_copy = tree.copy()
        dt = DecisionTree(v)
        for cc in tree_copy.ccs(v):
            dt_cc = None
            cc_hash = hash(cc)
            if dp[cc_hash] is not None:
                dt_cc = dp[cc_hash]
            else:
                dt_cc = DP(cc, dp)
            dt.attach_subtree(dt_cc, v)
        candidate_dts.append(dt)

    dc = min(candidate_dts, key=lambda dt: dt.cost())
    dp[hash(tree)] = dc
    return dc


def tree_search_cicalese(tree: Tree, t: int):
    if len(tree) <= t:
        return DP(tree)
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
        subtree = subtrees[len(subtrees)-1]
        set_X = tree.vertices_of_degree_at_least(3)

