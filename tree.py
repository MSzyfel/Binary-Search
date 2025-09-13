from __future__ import annotations

from itertools import combinations
import random
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import copy


class Tree(nx.DiGraph):
    def __init__(self, base: any = None, n: int = None, seed: int = None, distribution=None):
        """
        Initializes a random directed rooted tree with 'n' nodes and a random root.
        Each node is assigned a random attribute 'c'.

        Parameters:
            n (int): number of nodes (must be ≥ 1)
            seed (int, optional): random seed
            distribution (callable, optional): function returning a float for node attribute 'c'
                                               (default: uniform(0,1))
        """
        super().__init__()
        self.seed = None
        self.root = None
        self.rank = None
        if isinstance(base, int | list | dict | tuple):
            if isinstance(base, int | tuple):
                self.add_nodes_from([base])
            else:
                self.add_nodes_from(base)
            return
        super().__init__()
        if isinstance(base, nx.DiGraph | Tree):
            nodes = list(base.nodes.data())
            self.add_nodes_from(nodes)
            self.add_edges_from(base.edges(data=True))
            self.n = len(base)
            if self.n > 0:
                self.root = self.get_root()
            if isinstance(base, Tree):
                self.rank = base.rank
            return
        if n is None:
            return
        self.n = n

        rnd = random.Random(seed)
        if distribution is None:
            distribution = lambda: rnd.uniform(0, 1)

        if seed is None:
            seed = random.seed(None)
        # Generate a labeled rooted directed tree (arborescence)
        undirected_tree = nx.random_labeled_tree(n=n, seed=seed)
        root = rnd.choice(list(undirected_tree.nodes))
        self.root = root
        bfs_edges = list(nx.bfs_edges(undirected_tree, source=root))
        # Add nodes with attribute 'c'
        for node in undirected_tree.nodes:
            self.add_node(node, c=distribution())

            # Dodaj ukierunkowane krawędzie
        self.add_edges_from(bfs_edges)
        self.seed = seed

    def __copy__(self) -> Tree:
        return copy.deepcopy(self)

    def hash(self, labels):
        nodes = sorted(self.nodes)
        labels = sorted(labels)
        g_hash = 0
        node_iterator = 0
        for i in range(0, labels[len(labels) - 1]):
            if nodes[node_iterator] == labels[i]:
                g_hash = g_hash + 2 ** i
                node_iterator = node_iterator + 1
            if node_iterator == len(nodes):
                break
        return g_hash

    def vcost(self, v):
        if isinstance(v, tuple):
            v = v[0]
        if 'c' in self.nodes[v]:
            return self.nodes[v]['c']
        else:
            return 1

    def get_root(self):
        r = next(v for v in self.nodes() if self.in_degree(v) == 0)
        self.root = r
        return self.root

    def ccs(self, vertices) -> list[Tree]:

        G = nx.DiGraph(self)
        if not isinstance(vertices, set):
            if isinstance(vertices, tuple):
                vertices = vertices[0]
            vertices = {vertices}
        G.remove_nodes_from(vertices)
        undir_G_minus_v = G.to_undirected()

        subtrees = []
        for component in nx.connected_components(undir_G_minus_v):
            subgraph = Tree(copy.deepcopy(self.subgraph(component)))
            subtrees.append(subgraph)

        return subtrees

    def attach_subtree(self, other: Tree, v):
        if len(other) == 0:
            return
        root_other = other.get_root()
        if isinstance(v, tuple):
            v = v[0]
        self.add_nodes_from(other.nodes(data=True))
        self.add_edges_from(other.edges(data=True))
        self.add_edge(v, root_other)

    def centroid(self):
        v = self.get_root()
        while not self.is_centroid(v):
            v_ccs = self.ccs(v)
            max_cc = max(v_ccs, key=lambda cc: len(cc))
            v = self.find_neighbor(max_cc, v)
        return v

    def is_centroid(self, v):
        for cc in self.ccs(v):
            if len(cc) > len(self) / 2:
                return False
        return True

    def find_neighbor(self, subtree: Tree, v):
        for u in subtree.nodes():
            if nx.Graph(self).has_edge(v, u):
                return u
        return None

    def is_descendant(self, node, ancestor) -> bool:
        return nx.has_path(self, source=ancestor, target=node)

    def pairs_of_ordered_vertices(self, nodes: set):
        for u in nodes:
            for v in nodes:
                if u < v:
                    yield u, v

    def minimal_subtree(self, terminals: set) -> Tree:
        nodes = set()
        edges = set()

        if len(terminals) < 2:
            subtree = Tree()
            subtree.add_nodes_from(terminals)
            return subtree
        for u, v in self.pairs_of_ordered_vertices(terminals):
            path = nx.shortest_path(nx.Graph(self), source=u, target=v)
            nodes.update(path)
            for i in range(0, len(path) - 1):
                u = path[i]
                v = path[i + 1]
                if self.is_descendant(u, v):
                    edges.add((v, u))
                else:
                    edges.add((u, v))
        subgraph = self.subgraph(nodes)
        minimal_subtree = copy.deepcopy(Tree(subgraph))
        #
        return minimal_subtree

    def __lt__(self, other):
        return len(self) < len(other)

    def cost(self):
        return len(self)

    def vertices_of_degree_at_least(self, degree: int) -> set:
        vertices = set()
        for v in self.nodes():
            if self.degree(v) >= degree:
                vertices.add(v)
        return vertices

    def minimal_subtree_with_contracted_paths(self, terminals: set) -> Tree:
        nodes = set()
        edges = set()
        nodes = nodes | terminals
        for u, v in self.pairs_of_ordered_vertices(terminals):
            path_u_v = nx.shortest_path(nx.Graph(self), source=v, target=u)
            set_of_vertices_of_path_u_v = set(path_u_v)
            if terminals & set_of_vertices_of_path_u_v == {u, v}:
                if len(set_of_vertices_of_path_u_v) == 2:
                    self.add_rooted_edge(edges, u, v)
                else:
                    l = min(list(set_of_vertices_of_path_u_v - {u, v}), key=lambda v: self.vcost(v))
                    nodes.add(l)
                    self.add_rooted_edge(edges, u, l)
                    self.add_rooted_edge(edges, l, v)
        subtree = Tree(self.subgraph(nodes))
        subtree.add_edges_from(edges)
        return subtree

    def is_subtree(self, subtree: Tree) -> bool:
        return set(self.nodes).issubset(subtree.nodes) and set(self.edges).issubset(subtree.edges)

    def draw(self, orientation: str = 'vertical', attribute=None, type=None):
        nodes = self.nodes()
        if type == None:
            labels = {node: f"{node}" for node in nodes}
        if type == float:
            labels = {node: f"{node}: {self.nodes[node][attribute]:.2f}" for node in
                      nodes}  # etykiety z indeksem i wagą
        if type == int:
            labels = {node: f"{node}: {self.nodes[node][attribute]}" for node in nodes}
        source = self.get_root()
        pos = nx.bfs_layout(self, source, align=orientation)
        nx.draw(self, pos, with_labels=True, labels=labels, node_color='lightblue')
        plt.show()

    def add_rooted_edge(self, edges, u, v):
        if self.is_descendant(u, v):
            edges.add((v, u))
        else:
            edges.add((u, v))

    def parent(self, v):
        if self.get_root() == v:
            return None
        predecessors = list(self.predecessors(v))[0]
        return predecessors

    def neighboring_edges(self, v):
        return list(self.in_edges(v)) + list(self.out_edges(v))

    def contracted_heavy_groups(self, c, v: int = None, new_tree: Tree = None) -> Tree | None:
        nodes = self.nodes(data=True)
        if v is None:
            v = self.get_root()
            if nodes[v]['c'] > c:
                return None
        if new_tree is None:
            new_tree = Tree()
            new_tree.add_node(v, **nodes[v])
        for u in self.successors(v):
            w_u = nodes[u]['c']
            if w_u <= c:
                data = nodes[u]
                new_tree.add_node(u, **data)
                new_tree.add_edge(v, u)
                self.contracted_heavy_groups(c=c, v=u, new_tree=new_tree)
            else:
                queue = [u]
                while queue:
                    u = queue.pop(0)
                    for c in self.successors(u):
                        if nodes[c]['c'] <= c:
                            data = nodes[c]
                            new_tree.add_node(c, **data)
                            new_tree.add_edge(v, c)
                            self.contracted_heavy_groups(c=c, v=c, new_tree=new_tree)
                        else:
                            queue.append(c)

        return new_tree

    def reroot_by_min_attr(self, attr='c'):
        # 1. Znajdź nowy root na podstawie minimalnej wartości danego atrybutu
        new_root = min(self.nodes, key=lambda u: self.nodes[u].get(attr, float('inf')))

        # 2. Zainicjalizuj nowe drzewo i ustaw root
        new_tree = Tree()
        new_tree.root = new_root

        # 3. Przenieś atrybuty wierzchołków do nowego drzewa
        for node in self.nodes:
            new_tree.add_node(node, **self.nodes[node])  # zakładamy, że self.nodes[node] to dict atrybutów

        # 4. BFS od nowego roota, budujemy drzewo "w dół"
        visited = set()
        queue = [new_root]
        visited.add(new_root)

        while queue:
            u = queue.pop(0)

            # rozważamy tylko krawędzie prowadzące do wierzchołków, które nie były jeszcze odwiedzone
            for v in list(self.predecessors(u)) + list(self.successors(u)):
                if v not in visited:
                    new_tree.add_edge(u, v)
                    visited.add(v)
                    queue.append(v)

        return new_tree

    def round_values(self, round_function, attribute='c'):
        for node in self.nodes:
            if attribute in self.nodes[node]:
                original_value = self.nodes[node][attribute]
                new_value = round_function(original_value)
                self.nodes[node][attribute] = new_value

    def get_subtree(self, v):
        return copy.deepcopy(nx.dfs_tree(self, source=v))

    def get_neighbors(self, v):
        pred = list(self.predecessors(v))
        succ = list(self.successors(v))
        return pred + succ

    def fulfil_star_condition(self):
        excluded_queries = []
        for v in self.nodes():
            neighbors = self.get_neighbors(v)
            sum_of_neighbor_costs = sum(self.nodes[neighbor]['c'] for neighbor in neighbors)
            if self.nodes[v]['c'] > sum_of_neighbor_costs:
                excluded_queries.append(v)
                self.nodes[v]['c'] = sum_of_neighbor_costs
        return excluded_queries

    def get_heavy_groups(self, a):
        heavy_nodes = [v for v, data in self.nodes(data=True) if data.get('c', float('-inf')) > a]

        if not heavy_nodes:
            return []

        undir = nx.Graph(self)
        induced = undir.subgraph(heavy_nodes)

        groups: list[Tree] = []
        for comp in nx.connected_components(induced):
            subgraph = self.subgraph(comp)
            group_tree = Tree(copy.deepcopy(subgraph))
            groups.append(group_tree)

        return groups

    @classmethod
    def make_k_up_modular(cls, n: int, k: int,
                          seed: int | None = None) -> Tree:
        tree = Tree(n=n, seed=seed)
        nodes = list(tree.nodes(data=True))
        nodes.sort(key=lambda v: -v[1]['c'])
        for node, _ in nodes:
            s = tree.nodes[node]['c']
            hs = tree.get_heavy_groups(s)
            while len(hs) > k:
                h1 = hs[0]
                h2 = hs[1]
                hv1 = list(h1.nodes())[0]
                hv2 = list(h2.nodes())[0]
                path = tree.minimal_subtree({hv1, hv2})
                for v in path:
                    tree.nodes[v]['c'] = s
                hs = tree.get_heavy_groups(s)
        return tree

    def is_k_up_modular(self, k):
        for node in self.nodes():
            s = self.nodes[node]['c']
            hs = self.get_heavy_groups(s)
            ks = len(hs)
            if ks > k:
                return False
        return True
