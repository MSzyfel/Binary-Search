import heapq
from math import log2, ceil, floor, sqrt, log

import networkx as nx
from tree import Tree
from decision_tree import DecisionTree
from extended_dt import ExtendedDT
from max_heap_object import MaxHeapObj
import copy
from itertools import combinations
from separator import Separator

INF = float('inf')


def dp_tree(tree: Tree, nodes: list = None, dp=None):
    if dp is None:
        dp = {}
    if nodes is None:
        nodes = tree.nodes()
    if len(tree) == 1:
        vertices = list(tree.nodes(data=True))
        v = vertices[0]
        dt = DecisionTree(v)
        dp[tree.hash(nodes)] = copy.deepcopy(dt)
        return dt
    candidate_dts = []
    for v in tree.nodes(data=True):
        tree_copy = copy.deepcopy(tree)
        dt = DecisionTree(v)
        for cc in tree_copy.ccs(v):
            cc_hash = cc.hash(nodes)
            if cc_hash in dp:
                dt_cc = copy.deepcopy(dp[cc_hash])
            else:
                dt_cc = copy.deepcopy(dp_tree(cc, nodes, dp))
            dt.attach_subtree(dt_cc, v)
        candidate_dts.append(dt)

    dc = min(candidate_dts, key=lambda dt: dt.cost())
    dp[tree.hash(nodes)] = copy.deepcopy(dc)
    return dc

def dereniowski_inspired(tree: Tree) -> DecisionTree:
    t = int(2 ** (sqrt(log2(len(tree)))))
    dt1 = tree_search_cicalese_inspired(tree, t, qptas_dereniowski_inspired)
    return dt1

def cicalese_inspired(tree: Tree) -> DecisionTree:
    t = int(log2(len(tree)))
    dt1 = tree_search_cicalese_inspired(tree, t, dp_tree)
    return dt1


def tree_search_cicalese_inspired(tree: Tree, t: int, base_algorithm):
    if len(tree) <= t:
        return base_algorithm(tree)
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
        dt_y = base_algorithm(y)
        dt_y = DecisionTree(dt_y)
        y_nodes = set(dt_y.nodes())
        for p in tree_on_x.ccs(y_nodes):
            dt_p = dp_tree(p)
            dt_y.attach_sub_dt(tree_on_x, p, dt_p)
        for h in tree.ccs(set(tree_on_x.nodes())):
            dt_h = tree_search_cicalese_inspired(h, t, base_algorithm)
            dt_y.attach_sub_dt(tree, h, dt_h)
        return dt_y


def calculate_ranking(tree: Tree, r, gbs: dict[int: int], lbu: dict[int: int]) -> int:
    children = list(tree.children(r))
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
    n = 2 ** (ceil(log2(n))) + 1
    result = {0: 0}
    for k in range(0, n):
        gbs = k.bit_length() - 1
        result[k] = gbs
    return result


def calculate_lbu(n):
    n = 2 ** (ceil(log2(n))) + 1
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
    children = tree.children(v)
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
    tree.rank = gbs[sr]
    # q = {}
    # W = [(None, None)] * (tree.rank + 1)
    # visit(tree=tree, f_dash=INF, v=root, W=W, q=q)
    # qv = max(list(tree.nodes(data=True)), key=lambda v: v[1]['r'])[0]
    # visited_edges = set()
    # d = backtrack_dt(tree, qv, q, visited_edges)
    d = slow_backtrack_dt(tree)
    return d


const = 24
def qptas_dereniowski_inspired(tree: Tree, epsilon=const) -> DecisionTree:
    if len(tree) == 0:
        return DecisionTree()
    tree_unchanged = copy.deepcopy(tree)
    #tree.draw(attribute='c', type=float)
    tree = tree.reroot_by_min_attr('c')
    nodes = tree.nodes(data=True)
    max_val = max([node[1]['c'] for node in nodes])

    def round_function(value):
        return value / max_val

    tree.round_values(round_function)
    # excluded_queries = tree.fulfil_star_condition()

    p = ceil(const / epsilon)
    n = len(tree)
    ranking_based_dt(tree)
    r = tree.rank
    d = p*p*(r+1)
    step = 1 / (p * n)
    c = step
    step_index = 0.0
    dt = None
    while dt is None:
        step_index += 1
        c = step * step_index
        tree_copy = copy.deepcopy(tree)

        def round_function(value):
            return ceil_base(value, c) if value > p * c else ceil_base(value, 1 / (p * n))

        tree_copy.round_values(round_function)
        dt_a = dp_timelines_non_uni_costs(tree=tree_copy, p=p, c=c, n=n, depth=d)
        if dt_a is not None:
            contracted_tree = tree_copy.contracted_heavy_groups(p * c)
            if contracted_tree is not None:
                dt_lt = ranking_based_dt(contracted_tree)
                dt = merge_dts(tree_copy, dt_a, [dt_lt])
            else:
                dt = dt_a
                break

    for node in dt.nodes():
        dt.nodes[node]['c'] = tree_unchanged.nodes[node]['c']
    return dt


def ceil_base(value, base):
    val = base * ceil(value / base)
    return val


def merge_dts(tree: Tree, dt_a: DecisionTree, ft_lt: list[DecisionTree]) -> DecisionTree | None:
    if len(ft_lt) == 1:
        r = ft_lt[0].get_root()
        ft_lt = ft_lt[0].ccs(r)
    else:
        r = dt_a.get_root()
    dt = DecisionTree()
    dt.add_node(r)
    for h in tree.ccs(r):
        h=Tree(h)
        ft_lt_h = []
        for dt_lt in ft_lt:
            dt_lt = DecisionTree(dt_lt)
            if  set(dt_lt.nodes()).issubset(h.nodes()):
                ft_lt_h.append(dt_lt.restrict_to(h))
        dt_a_h = dt_a.restrict_to(h)
        dt_h = merge_dts(h, dt_a_h, ft_lt_h)
        dt.attach_sub_dt(tree, h, dt_h)
    return dt






# def build_dt_lt_root(tree: Tree, dt_lt: DecisionTree, p, c, n, depth) -> DecisionTree | None:
#     q = dt_lt.get_root()
#     dt = DecisionTree()
#     data = tree.nodes(data=True)[q]
#     dt.add_node(q, **data)
#     responses = dt_lt.query(tree)
#     for cc, dt_lt_cc in responses.items():
#         cc = Tree(cc)
#         dt_lt_cc = DecisionTree(dt_lt_cc)
#         r_cc = cc.get_root()
#         if cc.nodes[r_cc]['c'] > p * c:
#             dt_cc = build_dt_h_root(cc, dt_lt_cc, p, c, n, depth)
#         else:
#             dt_cc = build_dt_lt_root(cc, dt_lt_cc, p, c, n, depth)
#         if dt_cc is None:
#             return None
#         dt.attach_subtree(dt_cc, q)
#     # if dt.cost() > c * depth:
#     #     return None
#     return dt


# def build_dt_h_root(tree: Tree, dt_lt: DecisionTree, p, c, n, depth) -> DecisionTree | None:
#     child_dts = {}
#     subtree = tree.copy()
#     forbidden_directions = {tree.get_root()}
#     ccs_dt_lt = list(nx.weakly_connected_components(dt_lt))
#     for cc_dt_lt_nodes in ccs_dt_lt:
#         cc_dt_lt = dt_lt.subgraph(cc_dt_lt_nodes)
#         while len(cc_dt_lt) > 0:
#             responses = cc_dt_lt.query(subtree, excluded_responses=forbidden_directions)
#             cc_dd_lt_root = cc_dt_lt.get_root()
#             forbidden_directions.add(cc_dd_lt_root)
#             dt_lt.remove_node(cc_dd_lt_root)
#             if len(responses) > 0:
#                 child_dts[cc_dd_lt_root] = []
#                 for response_tree, response_dt_lt in responses.items():
#                     response_tree = Tree(response_tree)
#                     response_dt_lt = DecisionTree(response_dt_lt)
#                     subtree.remove_nodes_from(response_tree.nodes())
#                     dt_lt.remove_nodes_from(response_dt_lt.nodes())
#                     if response_tree.nodes[response_tree.get_root()]['c'] > p * c:
#                         response_dt = build_dt_h_root(response_tree, response_dt_lt, p, c, n, depth)
#                     else:
#                         response_dt = build_dt_lt_root(response_tree, response_dt_lt, p, c, n, depth)
#                     if response_dt is None:
#                         return None
#                     child_dts[cc_dd_lt_root].append(response_dt)
#     dt = dp_timelines_non_uni_costs(tree=subtree, child_dts=child_dts, p=p, c=c, n=n, depth=depth)
#     return dt


def dp_timelines_non_uni_costs(tree: Tree, p: int, c: float, n: int,
                               depth: int) -> DecisionTree | None:
    slot_size = (1 / (p * n))
    table = {}

    def retrieve(v, i, timeline) -> ExtendedDT:
        loads = timeline.get_loads()
        # for i in range(len(loads)):
        #     load1= round(loads[i][0],5)
        #     loads[i] = (load1, loads[i][1])
        if (v, i, str(loads)) == (35, 1, '[(0.0, False), (0.5714285714285714, True), (0.5714285714285714, False)]'):
            a = 0
        if (v, i, str(loads)) in table:
            return copy.deepcopy(table[(v, i, str(timeline.get_loads()))])
        else:
            dt = dp(v, i, timeline)
            table[(v, i, str(loads))] = dt
            return copy.deepcopy(dt)

    def dp(v: int, i: int, timeline: ExtendedDT) -> ExtendedDT | None:
        #print(str(c) + ", " + str(v) + ", " + str(i), str(timeline.get_loads()))
        if i == 0:
            return dp_no_children(v, timeline)
        if i == 1:
            candidate_dts = dp_one_child(v, timeline)
        else:
            candidate_dts = dp_many_children(v, i, timeline)
        if len(candidate_dts) == 0:
            return None
        result_dt = min(candidate_dts, key=lambda dt: dt.cost())
        return result_dt

    def dp_no_children(v: int, timeline: ExtendedDT) -> ExtendedDT | None:
        cost = tree.vcost(v)
        is_heavy = cost > p * c
        for box_index in range(depth):
            for slot_index in range(int(round(c / slot_size, 0))) if not is_heavy else [0]:
                try:
                    dt = copy.deepcopy(timeline)
                    dt.put_query(box_index=box_index, slot_index=slot_index, query=v, cost=cost,
                                 is_heavy=is_heavy, is_first=False)
                    if dt.cost() <= depth * c:
                        return dt
                    else:
                        return None
                except:
                    continue
        return None

    def dp_one_child(v: int, timeline: ExtendedDT) -> list[ExtendedDT | None]:
        candidate_dts = []
        cost = tree.vcost(v)
        is_heavy = cost > p * c
        for box_index in range(depth):
            for slot_index in range(int(round(c / slot_size, 0))) if not is_heavy else [0]:
                try:
                    dtb = copy.deepcopy(timeline)
                    dtb.put_query(box_index=box_index, slot_index=slot_index, query=v, cost=cost,
                                  is_heavy=is_heavy, is_first=False)
                    dt_cost = dtb.cost()
                    if dt_cost <= depth * c:
                        loads = dtb.get_loads()
                        occupied_boxes = ceil((cost + slot_index * slot_size) / c)
                        for box_below in range(box_index + occupied_boxes, depth):
                            loads[box_below] = (0.0, False)
                        new_timeline = ExtendedDT(base=loads, max_depth=depth, box_size=c, slot_size=slot_size)
                        u = tree.children(v)[0]
                        outdegree = len(tree.children(u))
                        dtb2 = retrieve(v=u, i=outdegree, timeline=new_timeline)
                        if dtb2 is None:
                            continue
                        if v == 23 and box_index == 0:
                            a = 0
                        dtb2loads = dtb2.get_loads()
                        dtb.merge(other=dtb2, box=box_index + occupied_boxes - 1, query=v)
                        dtbcost = dtb.cost()
                        if dtbcost <= depth * c:
                            candidate_dts.append(dtb)
                except:
                    continue
        return candidate_dts

    def dp_many_children(v: int, i: int, timeline: ExtendedDT) -> list[ExtendedDT | None]:
        candidate_dts = []
        for loads1, loads2 in timeline.all_bipartitions():
            new_timeline = ExtendedDT(base=loads1, box_size=c, max_depth=depth, slot_size=slot_size)
            loads = new_timeline.get_loads()
            dt1 = retrieve(v=v, i=i - 1, timeline=new_timeline)
            if dt1 is None:
                continue
            v_box_index = None
            for j in range(depth):
                if v in dt1.query_sequence(j):
                    v_box_index = j
            #loads2[v_box_index] = (loads2[v_box_index][0], False)
            for j in range(v_box_index + 1, depth):
                loads2[j] = (0.0, False)
            new_timeline = ExtendedDT(base=loads2, box_size=c, max_depth=depth, slot_size=slot_size)
            children = tree.children(v)
            u = children[i - 1]
            outdegree = len(tree.children(u))
            dt2 = retrieve(v=u, i=outdegree, timeline=new_timeline)
            if dt2 is None:
                continue
            dt1.merge(other=dt2, box=v_box_index, query=v)
            if dt1.cost() <= depth * c:
                candidate_dts.append(dt1)
        return candidate_dts

    root = tree.get_root()
    timeline = ExtendedDT(box_size=c, max_depth=depth, slot_size=slot_size)
    successors = list(tree.children(root))
    ext_dt = dp(v=root, i=len(successors), timeline=timeline)
    if ext_dt is not None:
        dt = ext_dt.to_decision_tree()
        return dt
    else:
        return None


def k_up_modularity_algorithm(tree: Tree):
    n = len(tree)
    if n < 4:
        t = 1
    else:
        t = int(2 ** (sqrt(log2(n))))

    def create_dt(current_tree: Tree, a: float, b: float):
        if b <= 1 / log2(n) or all(tree.vcost(v) > a for v in tree.nodes()):
            dt = ranking_based_dt(current_tree)
            return dt
        elif len(current_tree) <= t:
            dt = qptas_dereniowski_inspired(current_tree)
            return dt
        else:
            x = set()
            for h in current_tree.get_heavy_groups(a):
                heavy_group = list(h.nodes())
                v = heavy_group[0]
                x.add(v)
            tree_on_x = tree.minimal_subtree(x)
            y = x | tree_on_x.vertices_of_degree_at_least(3)
            t_z = tree.minimal_subtree_with_contracted_paths(y)
            z = set(t_z.nodes())
            d = qptas_dereniowski_inspired(t_z)
            for ccs in current_tree.ccs(z):
                hs = ccs.get_heavy_groups(a)
                if len(hs) == 1:
                    h = hs[0]
                    d_h = ranking_based_dt(h)
                    d.attach_sub_dt(current_tree, ccs, d_h)
                    for l_ccs in ccs.ccs(set(h.nodes())):
                        d_l = create_dt(l_ccs, a / 2, a)
                        d.attach_sub_dt(current_tree, l_ccs, d_l)
                else:
                    d_l = create_dt(ccs, a / 2, a)
                    d.attach_sub_dt(current_tree, ccs, d_l)
            return d

    if n == 1:
        dt = DecisionTree()
        node_id, attrs = list(tree.nodes(data=True))[0]
        dt.add_node(node_id, **attrs)
        return dt
    if n < 4:
        a = 0
    else:
        num = 2 ** (ceil(log2(log2(n))) - 1)
        denom = log2(n)
        a = num / denom

    d = create_dt(tree, a, 1.0)
    d.append_costs(tree)
    return d


def separator(tree, alpha):
    w_tree = tree.sum_of('w')
    k = floor(w_tree / alpha)
    table = {}

    nodes = tree.nodes(data=True)

    def retrieve(v, i=None, is_in=None, w=None) -> Separator:
        if (v, is_in, i, w) in table:
            sep = table[(v, is_in, i, w)]
        else:
            sep = dp(v, is_in, i, w)
            table[(v, is_in, i, w)] = sep
        return sep

    def dp(v, is_in=None, i=None, w=None) -> Separator:
        successors = tree.children(v)
        if is_in is None:
            options = [retrieve(v=v, is_in=True)] + [retrieve(v=v, is_in=False, w=w) for w in range(0, k + 1)]
        elif is_in is True:
            if len(successors) == 0:
                sep = Separator({v: nodes[v]})
            else:
                sep = Separator([{v: nodes[v]}] + [retrieve(c) for c in successors])
            return sep
        else:
            if i is None:
                return retrieve(v=v, is_in=is_in, i=len(successors), w=w)
            if i == 0:
                sep = Separator()
                if not w == tree.nodes[v]['w'] <= k:
                    sep.cost = INF
                return sep
            if i == 1:
                w_v = tree.nodes[v]['w']
                c = successors[0]
                if w < w_v:
                    sep = Separator(set())
                    sep.cost = INF
                    return sep
                elif w == w_v:
                    options = [retrieve(v=c, is_in=True)] + [retrieve(v=c, is_in=False, w=0)]

                else:
                    return retrieve(v=c, is_in=False, w=w - w_v)
            else:
                c_i = successors[i - 1]
                options = [Separator([retrieve(v, i - 1, is_in=False, w=w), retrieve(c_i, is_in=True)])] + [Separator(
                    [retrieve(v, i - 1, is_in=False, w=w - j), retrieve(c_i, is_in=False, w=j)]) for j in
                    range(0, w + 1)]
        sep = min(options, key=lambda option: option.cost)
        return sep

    sep = dp(tree.get_root())
    return sep


def separatorFPTAS(tree: Tree, alpha, delta):
    n = len(tree)
    w_tree = tree.sum_of('w')
    K = delta * w_tree / (n * alpha)
    prim_tree = tree.copy()
    for node in prim_tree.nodes():
        new_w = floor(prim_tree.nodes[node]['w'] / K)
        prim_tree.nodes[node]['w'] = new_w
    w_prim_tree = prim_tree.sum_of('w')
    alpha_prim = alpha * K * w_prim_tree / w_tree
    # prim_tree.draw(attribute='w', type=float)
    sep = separator(prim_tree, alpha_prim)
    for node in sep.vset:
        new_w = tree.nodes[node]['w']
        sep.vset[node]['w'] = new_w
    return sep


def build_any_dt(tree: Tree, sep: Separator):
    dt = DecisionTree()
    for v, data in sep.vset.items():
        v_data = {v: data}
        htv = Tree(v_data)
        dtv = DecisionTree(v_data)
        dt.attach_sub_dt(tree=tree, h=htv, dt_h=dtv)
    return dt


def average_case_trees_fptas(tree: Tree, epsilon=1) -> DecisionTree:
    sep = separatorFPTAS(tree, alpha=2, delta=epsilon / (4 + epsilon))
    dt = build_any_dt(tree=tree, sep=sep)
    for h in tree.ccs(set(sep.vset.keys())):
        dt_h = average_case_trees_fptas(tree=h, epsilon=epsilon)
        dt.attach_sub_dt(tree=tree, h=h, dt_h=dt_h)
    return dt


def centroid_dt(tree: Tree):
    c = tree.centroid(weighted=True)
    c_data = tree.nodes(data=True)[c]
    dt = DecisionTree({c: c_data})
    for ccs in tree.ccs(c):
        dt_ccs = centroid_dt(ccs)
        dt.attach_sub_dt(tree=tree, h=ccs, dt_h=dt_ccs)
    return dt


def average_case_PTAS(tree: Tree, epsilon=1):
    n = len(tree)
    depth = ceil((1 + 1 / epsilon) * ceil(log2(n)))
    dt = dp_timelines_uni_costs(tree, depth, 'average')
    return dt


def average_case_FPTAS(tree: Tree, epsilon=1):
    n = len(tree)
    w_tree = tree.sum_of('w')
    K = epsilon * w_tree / (n*min(n, ceil(w_tree*max(1,ceil(log(w_tree, 1.5)))))) # check if the alternative lower bound is correct
    prim_tree = tree.copy()
    for node in prim_tree.nodes():
        new_w = ceil(prim_tree.nodes[node]['w'] / K)
        prim_tree.nodes[node]['w'] = new_w
    w_prim_tree = prim_tree.sum_of('w')
    depth = max(1,ceil(log(w_prim_tree, 1.5)))
    print(depth)
    #prim_tree.draw(attribute='w', type=float)
    dt = dp_timelines_uni_costs(prim_tree, depth, 'average')
    for node in dt.nodes():
        new_w = tree.nodes[node]['w']
        dt.nodes[node]['w'] = new_w
    return dt


def dp_timelines_uni_costs(tree, depth, crit) -> DecisionTree:
    table = {}
    counter = 0

    def retrieve(v, i, timeline) -> ExtendedDT:
        nonlocal counter
        loads = timeline.get_loads()
        loads = [int(load[0]) for load in loads]
        if (v, i, str(loads)) in table:
            return table[(v, i, str(loads))]
        else:
            counter +=1
            #print(counter)
            dt = dp(v, i, timeline)
            table[(v, i, str(loads))] = dt
            return dt

    def dp(v: int, i: int, timeline: ExtendedDT) -> ExtendedDT | None:
        #print(str(v) + ", " + str(i), str(timeline.get_loads()))
        if i == 0:
            return dp_no_children(v, timeline)
        if i == 1:
            candidate_dts = dp_one_child(v, timeline)
        else:
            candidate_dts = dp_many_children(v, i, timeline)
        if len(candidate_dts) == 0:
            return None
        result_dt = min(candidate_dts, key=lambda dt: dt.cost(crit))
        return result_dt

    def dp_no_children(v: int, timeline: ExtendedDT) -> ExtendedDT | None:
        for box_index in range(depth):
            try:
                dt = copy.deepcopy(timeline)
                dt.put_query(box_index=box_index, slot_index=0, query=v, cost=1, weight = tree.nodes[v]['w'],
                             is_heavy=True, is_first=False)
                return dt
            except:
                continue
        return None

    def dp_one_child(v: int, timeline: ExtendedDT) -> list[ExtendedDT | None]:
        candidate_dts = []
        for box_index in range(depth):
            try:
                dtb = copy.deepcopy(timeline)
                dtb.put_query(box_index=box_index, slot_index=0, query=v, cost=1,
                              is_heavy=True, is_first=False, weight = tree.nodes[v]['w'])
                loads = dtb.get_loads()
                for box_below in range(box_index + 1, depth):
                    loads[box_below] = (0, False)
                new_timeline = ExtendedDT(base=loads, max_depth=depth)
                u = tree.children(v)[0]
                outdegree = len(tree.children(u))
                dtb2 = retrieve(v=u, i=outdegree, timeline=new_timeline)
                if dtb2 is None:
                    continue
                dtb.merge(other=dtb2, box=box_index, query=v)
                candidate_dts.append(dtb)
            except:
                continue
        return candidate_dts

    def dp_many_children(v: int, i: int, timeline: ExtendedDT) -> list[ExtendedDT | None]:
        candidate_dts = []
        for loads1, loads2 in timeline.all_bipartitions(trans = False):
            new_timeline = ExtendedDT(base=loads1, max_depth=depth)
            dt1 = retrieve(v=v, i=i - 1, timeline=new_timeline)
            if dt1 is None:
                continue
            v_box_index = None
            for j in range(depth):
                if v in dt1.query_sequence(j):
                    v_box_index = j
            for j in range(v_box_index + 1, depth):
                loads2[j] = (0, False)
            new_timeline = ExtendedDT(base=loads2, max_depth=depth)
            u = tree.children(v)[i - 1]
            outdegree = len(tree.children(u))
            dt2 = retrieve(v=u, i=outdegree, timeline=new_timeline)
            if dt2 is None:
                continue
            dt1.merge(other=dt2, box=v_box_index, query=v)
            candidate_dts.append(dt1)
        return candidate_dts

    root = tree.get_root()
    timeline = ExtendedDT(max_depth=depth)
    successors = list(tree.children(root))
    ext_dt = dp(v=root, i=len(successors), timeline=timeline)
    if ext_dt is not None:
        dt = ext_dt.to_decision_tree()
        return dt
    else:
        return None
