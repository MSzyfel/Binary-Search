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
        Each node is assigned a random attribute 'w'.

        Parameters:
            n (int): number of nodes (must be ≥ 1)
            seed (int, optional): random seed
            distribution (callable, optional): function returning a float for node attribute 'w'
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
            self.root = self.get_root()
            if isinstance(base, Tree):
                self.rank = base.rank
            return
        self.n = n
        if n is None:
            return

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
        # Add nodes with attribute 'w'
        for node in undirected_tree.nodes:
            self.add_node(node, w=distribution())

            # Dodaj ukierunkowane krawędzie
        self.add_edges_from(bfs_edges)
        self.seed = seed

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

    def weight(self, v):
        if isinstance(v, tuple):
            v = v[0]
        if 'w' in self.nodes[v]:
            return self.nodes[v]['w']
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
            return Tree(nx.DiGraph(terminals))
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

        return copy.deepcopy(self.edge_subgraph(edges))

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
                    l = min(list(set_of_vertices_of_path_u_v - {u, v}), key=lambda v: self.weight(v))
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

    def contracted_heavy_groups(self, w) -> Tree:
        pass

    def reroot_by_min_attr(self, attr='w'):
        # 1. Znajdź nowy root
        new_root = min(self.nodes, key=lambda u: self.nodes[u].get(attr, float('inf')))

        # 2. Zbuduj nowe drzewo jako BFS z nowego roota
        new_tree = Tree()
        new_tree.root = new_root

        visited = set()
        queue = [new_root]
        visited.add(new_root)

        while queue:
            u = queue.pop(0)
            for v in self.successors(u):
                if v not in visited:
                    continue  # ignorujemy krawędzie wychodzące, bo chcemy tylko dzieci w BFS z nowego roota
            for v in self.predecessors(u):
                if v not in visited:
                    new_tree.add_edge(u, v)  # od nowego roota w dół
                    visited.add(v)
                    queue.append(v)

        return new_tree

    def round_values(self, round_function, attribute='w'):
        for node in self.nodes:
            if attribute in self.nodes[node]:
                original_value = self.nodes[node][attribute]
                new_value = round_function(original_value)
                self.nodes[node][attribute] = new_value

