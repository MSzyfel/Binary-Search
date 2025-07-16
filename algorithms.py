import heapq
from math import log2, ceil

import numpy as np
import networkx as nx
from tree import Tree
from decision_tree import DecisionTree
from max_heap_object import MaxHeapObj
import copy

INF = float('inf')


def dp_tree(tree: Tree, nodes: list = None, dp=None):
    if dp is None:
        dp = {}
    if nodes is None:
        nodes = tree.nodes()
    if len(tree) == 1:
        nodes = list(tree.nodes(data=True))
        v = nodes[0]
        return DecisionTree(v)
    candidate_dts = []
    for v in tree.nodes(data=True):
        tree_copy = copy.deepcopy(tree)
        dt = DecisionTree(v)
        for cc in tree_copy.ccs(v):
            cc_hash = cc.hash(nodes)
            if cc_hash in dp:
                dt_cc = copy.deepcopy(dp[cc_hash])
            else:
                dt_cc = dp_tree(cc, nodes, dp)
            dt.attach_subtree(dt_cc, v)
        candidate_dts.append(dt)

    dc = min(candidate_dts, key=lambda dt: dt.cost())
    dp[tree.hash(nodes)] = copy.deepcopy(dc)
    return dc


def tree_search_cicalese_inspired(tree: Tree, t: int):
    if len(tree) <= t:
        return dp_tree(tree)
    else:
        centroids = set()
        subtrees = [MaxHeapObj(copy.deepcopy(tree))]
        for _ in range(0, t):
            if len(subtrees) == 0:
                break
            subtree = heapq.heappop(subtrees).val
            centroid = subtree.centroid()
            centroids.add(centroid)
            c_ccs = subtree.ccs(centroid)
            for cc in c_ccs:
                heapq.heappush(subtrees, MaxHeapObj(cc))

        tree_on_x = tree.minimal_subtree(centroids)
        x = centroids | tree_on_x.vertices_of_degree_at_least(3)
        y = tree.minimal_subtree_with_contracted_paths(x)
        dt_y = dp_tree(y, y.nodes())
        y_nodes = set(dt_y.nodes())
        for p in tree_on_x.ccs(y_nodes):
            dt_p = dp_tree(p)
            dt_y.attach_sub_dt(tree, p, dt_p)
        for h in tree.ccs(set(tree_on_x.nodes())):
            dt_h = tree_search_cicalese_inspired(h, t)
            dt_y.attach_sub_dt(tree, h, dt_h)
        return dt_y


def calculate_ranking(tree: Tree, r, gbs: dict[int: int]) -> int:
    children = list(tree.successors(r))
    s = [calculate_ranking(tree, child, gbs) for child in children]
    k = len(s)
    a = [0] * (k + 1)
    b = [0] * (k + 1)
    for i in range(1, k + 1):
        a[i] = a[i - 1] & s[i]
        b[i] = b[i - 1] & (a[i - 1] | s[i])
    m = gbs[b[k]]
    mask = lb_mask(m)
    ak_masked = a[k] | mask
    rank = gbs[ak_masked]
    tree.nodes[r]['r'] = rank
    n = len(tree)
    sr = (a[k] | (1 << m)) & (lb_mask(n - rank) << m)
    tree.nodes[r]['s'] = sr
    return sr


def calculate_gbs(n):
    n = 2 ** (ceil(log2(n)))
    result = {}
    for k in range(0, n):
        x = k + 1
        pos = (x & -x).bit_length() - 1
        result[k] = pos
    return result


def bit_indices(x: int):
    """Zwraca iterator po indeksach bitów ustawionych na 1 w x (od najmłodszego bitu)."""
    i = 0
    while x:
        if x & 1:
            yield i
        x >>= 1
        i += 1


def lb_mask(length):
    return (1 << (length + 1)) - 1


def visit(tree: Tree, v: int, f_dash, W: list, q: dict):
    def memorize_W(upto):
        # memorize W[0..upto] copy
        return W[:upto + 1]

    def restore_W(memo, upto):
        for i in range(upto + 1):
            W[i] = memo[i]

    vr = tree.nodes[v]['r']
    memo = memorize_W(vr)
    for i in range(vr + 1):
        W[i] = (None, None)
    children = list(tree.successors(v))
    for w in children:
        w_rank = tree.nodes[w]['r']
        W[w_rank] = (v, (v, w))
        sw = tree.nodes[w]['s']
        lb_1_sw = sw & (-sw)
        if lb_1_sw < vr:
            mask = lb_mask(vr)
            sw_masked = sw & mask
            gb_1_sw = sw_masked.bit_length() - 1
            q[(v, w)] = W[gb_1_sw][0]
            for x in bit_indices(sw_masked):
                W[x] = (None, None)
    restore_W(memo, vr)
    if f_dash != INF:
        W[vr] = (v, (v, tree.parent(v)))
    sv = tree.nodes[v]['s']
    if f_dash == INF:
        sv_masked = sv
    else:
        mask = lb_mask(f_dash)
        sv_masked = sv & mask
    for y in bit_indices(sv_masked):
        mask = lb_mask(y)
        sv_masked2 = sv_masked & mask
        for x in bit_indices(sv_masked2):
            if all(i not in sv for i in range(x + 1, y)):
                q[W[y][1]] = W[x][0]


def backtrack_dt(tree: Tree, qv, q) -> DecisionTree:
    d = DecisionTree(qv)
    for edge in tree.neighboring_edges(qv):
        if edge in q.keys():
            w = q[edge]
        else:
            u, v = edge
            w = q[(v, u)]
        d_w = backtrack_dt(tree, w, q)
        d.attach_subtree(d_w, qv)
    return d


def calculate_unweighted_dt(tree: Tree):
    n = len(tree)
    gbs = calculate_gbs(n)
    root = tree.get_root()
    calculate_ranking(tree, root, gbs)
    q = {}
    W = [(None, None)] * (tree.rank + 1)
    visit(tree=tree, f_dash=INF, v=root, W=W, q=q)
    qv = max(list(tree.nodes(data=True)), key=lambda v: v['r'])
    d = backtrack_dt(tree, qv, q)
    return d
