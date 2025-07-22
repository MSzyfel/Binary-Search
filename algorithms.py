import heapq
from math import log2, ceil

import numpy as np
import networkx as nx
from tree import Tree
from decision_tree import DecisionTree
from partial_dt import PartialDT
from max_heap_object import MaxHeapObj
import copy
from itertools import combinations

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


def calculate_ranking(tree: Tree, r, gbs: dict[int: int], lbu: dict[int: int]) -> int:
    children = list(tree.successors(r))
    k = len(children)
    if k == 0:
        tree.nodes[r]['r'] = 0
        tree.nodes[r]['s'] = 1
        return 1
    s = [calculate_ranking(tree, child, gbs, lbu) for child in children]
    a = [0] * (k + 1)
    b = [0] * (k + 1)
    for i in range(1, k + 1):
        a[i] = a[i - 1] | s[i - 1]
        b[i] = b[i - 1] | (a[i - 1] & s[i - 1])
    m = gbs[b[k]]
    mask = lb_mask(m)
    ak_masked = a[k] | mask
    rank = lbu[ak_masked]
    tree.nodes[r]['r'] = rank
    sr = a[k]
    sr = sr | (1 << (rank))
    mask = ~(lb_mask(rank - 1))
    sr = sr & mask
    tree.nodes[r]['s'] = sr
    return sr


def calculate_gbs(n):
    n = 2 ** (ceil(log2(n)))
    result = {0: 0}
    for k in range(0, n):
        gbs = k.bit_length() - 1
        result[k] = gbs
    return result


def calculate_lbu(n):
    n = 2 ** (ceil(log2(n)))
    result = {}
    for k in range(0, n):
        lbu = (~k) & (k + 1)
        result[k] = lbu.bit_length() - 1
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
    return (1 << length + 1) - 1


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
        W[vr] = (v, (v, w))
        visit(tree, w, vr, W, q)
        sw = tree.nodes[w]['s']
        lb_1_sw = (sw & -sw).bit_length() - 1
        if lb_1_sw < vr:
            mask = lb_mask(vr - 1)
            sw_masked = sw & mask
            gb_1_sw = sw_masked.bit_length() - 1
            q[(v, w)] = W[gb_1_sw][0]
            for x in bit_indices(sw_masked):
                W[x] = (None, None)
    restore_W(memo, vr)
    if f_dash != INF:
        W[vr] = (v, (v, tree.parent(v)))
    sv = tree.nodes[v]['s']
    bits = bit_indices(sv)
    for x, y in combinations(bits, 2):
        if x < y < f_dash and all(i not in sv for i in range(x + 1, y)):
            q[W[y][1]] = W[x][0]


def backtrack_dt(tree: Tree, qv, q, visited_edges) -> DecisionTree:
    d = DecisionTree(qv)
    for edge in tree.neighboring_edges(qv):
        if edge not in visited_edges:
            if edge not in q.keys():
                edge = (edge[1], edge[0])
            w = q[edge]
            visited_edges.add(edge)
            d_w = backtrack_dt(tree, w, q, visited_edges)
            d.attach_subtree(d_w, qv)
    return d


def slow_backtrack_dt(tree) -> DecisionTree:
    v = max(list(tree.nodes(data=True)), key=lambda v: v[1]['r'])[0]
    d = DecisionTree(v)
    for cc in tree.ccs(v):
        d_cc = slow_backtrack_dt(cc)
        d.attach_subtree(d_cc, v)
    return d


def ranking_based_dt(tree: Tree):
    n = len(tree)
    gbs = calculate_gbs(n)
    lbu = calculate_lbu(n)
    root = tree.get_root()
    sr = calculate_ranking(tree, root, gbs, lbu)
    tree.draw(attribute='s', type=int)
    tree.rank = gbs[sr]
    # q = {}
    # W = [(None, None)] * (tree.rank + 1)
    # visit(tree=tree, f_dash=INF, v=root, W=W, q=q)
    # qv = max(list(tree.nodes(data=True)), key=lambda v: v[1]['r'])[0]
    # visited_edges = set()
    # d = backtrack_dt(tree, qv, q, visited_edges)
    d = slow_backtrack_dt(tree)
    return d


def qptas_dereniowski_inspired(tree: Tree, epsilon=1) -> DecisionTree:
    tree = tree.reroot_by_min_attr('w')
    c = ceil(168 / epsilon)
    n = len(tree)
    step = 1 / (c * n)
    w = step
    dt = build_strategy(copy.deepcopy(tree), c, w, n)
    while dt is not None:
        w += step
        dt = build_strategy(tree, c, w, n)
    return dt


def ceil_base(value, base):
    return base * ceil(value / base)


def build_strategy(tree: Tree, c, w, n) -> DecisionTree | None:
    contracted_tree = tree.contracted_heavy_groups(c * w)
    dt_lt = ranking_based_dt(contracted_tree)

    def round_function(value):
        return ceil_base(value, w) if value > c * w else ceil_base(value, 1 / (c * n))

    tree.round_values(round_function)
    depth = c * c * ceil(log2(n))
    dt = build_dt_lt_root(tree, dt_lt, c, w, n, depth)
    return dt


def build_dt_lt_root(tree: Tree, dt_lt: DecisionTree, c, w, n, depth) -> DecisionTree | None:
    q = dt_lt.get_root()
    dt = DecisionTree(q)
    responses = dt_lt.query(tree)
    for cc, dt_lt_cc in responses.items():
        cc = Tree(cc)
        dt_lt_cc = DecisionTree(dt_lt_cc)
        r_cc = cc.get_root()
        if cc.nodes(data=True)[r_cc][1]['w'] > c * w:
            dt_cc = build_dt_h_root(cc, dt_lt_cc, c, w, n, depth)
        else:
            dt_cc = build_dt_lt_root(cc, dt_lt_cc, c, w, n, depth)
        if dt_cc is None:
            return None
        dt.attach_subtree(dt_cc, q)
    if dt.cost() > w * depth:
        return None
    return dt


def build_dt_h_root(tree: Tree, dt_lt: DecisionTree, c, w, n, depth) -> DecisionTree | None:
    child_dts = {}
    subtree = tree.copy()
    for cc_dt_lt in nx.weakly_connected_components(dt_lt):
        cc_dt_lt = DecisionTree(cc_dt_lt)
        while len(dt_lt) > 0:
            responses = cc_dt_lt.query(subtree, cc_dt_lt.nodes())
            if len(responses) > 0:
                cc_dd_lt_root = cc_dt_lt.get_root()
                child_dts[cc_dd_lt_root] = []
                for response_tree, response_dt_lt in responses.keys():
                    response_tree = Tree(response_tree)
                    response_dt_lt = DecisionTree(response_dt_lt)
                    subtree.remove_nodes_from(response_tree.nodes())
                    if response_tree.get_root()[1]['w'] > c * w:
                        response_dt = build_dt_h_root(response_tree, response_dt_lt, c, w, n, depth)
                    else:
                        response_dt = build_dt_lt_root(response_tree, response_dt_lt, c, w, n, depth)
                    if response_dt is None:
                        return None
                    child_dts[cc_dd_lt_root].append(response_dt)
    root = subtree.get_root()
    timeline = PartialDT(box_size=w, max_depth=depth)
    partial_dt = dp_timelines(subtree, root, len(subtree.successors(root)), timeline, child_dts, c, w, n, depth)
    if partial_dt is not None:
        dt = partial_dt.to_decision_tree()
        return dt
    else:
        return None


def dp_timelines(tree, v, i, timeline, child_dts, c, w, n, depth) -> PartialDT | None:
    pass
