from __future__ import annotations

from itertools import combinations, product
import random
import time
from math import log2

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import copy


class Tree(nx.DiGraph):
    def __init__(self, base: any = None, n: int = None, seed: int = None, distribution_c=None, distribution_w=None):
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
            elif isinstance(base, dict):
                self.add_nodes_from(base.items())
            else:
                self.add_nodes_from(base)
            return
        #super().__init__()
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
        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)
        if seed is None:
            seed = random.seed(None)
        # Generate a labeled rooted directed tree (arborescence)
        undirected_tree = nx.random_labeled_tree(n=n, seed=seed)
        root = rnd.choice(list(undirected_tree.nodes))
        self.root = root
        bfs_edges = list(nx.bfs_edges(undirected_tree, source=root))
        # Add nodes with attribute 'c'
        sum_w = 0
        for node in undirected_tree.nodes:
            w = distribution_w()
            self.add_node(node, c=distribution_c(), w=w)
            sum_w += w
        for node in self.nodes:
            self.nodes[node]['w'] = self.nodes[node]['w'] / sum_w
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

    def centroid(self, weighted=False):
        v = self.get_root()
        while not self.is_centroid(v, weighted):
            v_ccs = self.ccs(v)
            max_cc = max(v_ccs, key=lambda cc: cc.sum_of('w') if weighted else len(cc))
            v = self.find_neighbor(max_cc, v)
        return v

    def is_centroid(self, v, weighted=False):
        for cc in self.ccs(v):
            cc_sum = cc.sum_of('w')
            self_sum = self.sum_of('w') / 2
            if (weighted and cc_sum > self_sum) or ((not weighted) and len(cc) > len(self) / 2):
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

    def to_pdf(self, filename: str, orientation: str = 'vertical', attribute=None, type=None, proportions=(6, 8)):
        nodes = self.nodes()
        n = len(nodes)
        s = 150000 / (n * log2(n))
        f = (s ** (1 / 2)) / 5

        # --- etykiety w zależności od typu atrybutu ---
        if type is None:
            labels = {node: f"{node}" for node in nodes}
        elif type == float:
            labels = {node: f"{node}: {self.nodes[node][attribute]:.2f}" for node in nodes}
        elif type == int:
            labels = {node: f"{node}: {self.nodes[node][attribute]}" for node in nodes}
        else:
            raise ValueError("type must be None, int or float")

            # --- pozycjonowanie wierzchołków ---
        source = self.get_root()
        pos = nx.bfs_layout(self, source, align=orientation)

        # --- przygotowanie rysunku ---
        plt.figure(figsize=proportions)
        nx.draw(
            self,
            pos,
            with_labels=True,
            labels=labels,
            node_color='white',
            font_size=f,  # <-- mniejszy rozmiar etykiet
            font_weight='regular',
            node_size=s,  # opcjonalnie: wielkość węzłów
            edgecolors='black'
        )

        # --- dopilnowanie rozszerzenia .pdf ---
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        # --- zapis do PDF ---
        plt.tight_layout()
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()

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
        T = Tree(nx.dfs_tree(self, source=v))

        # Skopiuj atrybuty węzłów
        for node in T.nodes():
            if node in self.nodes:
                T.nodes[node].update(self.nodes[node])

        # Skopiuj atrybuty grafu (np. nazwa, typ, opis)
        T.graph.update(self.graph)

        return T

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
    def is_k_up_modular(self, k):
        for node in self.nodes():
            s = self.nodes[node]['c']
            hs = self.get_heavy_groups(s)
            ks = len(hs)
            if ks > k:
                return False
        return True

    def sum_of(self, f):
        nodes = self.nodes(data=True)
        return sum(node[1][f] for node in nodes)

    def children(self, v):
        return sorted(list(self.successors(v)))

    @staticmethod
    def _normalize_w(tree):
        total = sum(tree.nodes[v]['w'] for v in tree.nodes)
        for v in tree.nodes:
            tree.nodes[v]['w'] /= total

    @classmethod
    def random_bounded_degree(cls, n: int, delta: int,
                              seed: int | None = None,
                              distribution_c=None,
                              distribution_w=None) -> Tree:
        """
        Generate a random directed rooted tree with n nodes and maximum degree ≤ delta.

        Parameters
        ----------
        n : int
            Number of nodes.
        delta : int
            Maximum degree (total degree ≤ delta).
        seed : int, optional
            Random seed.
        distribution_c : callable, optional
            Function returning random float for 'c' attribute.
        distribution_w : callable, optional
            Function returning random float for 'w' attribute.

        Returns
        -------
        Tree
            Randomly generated tree with degree ≤ delta.
        """
        assert n >= 1, "Tree must have at least one node"
        assert delta >= 2, "Delta must be at least 2 (otherwise cannot form a tree with >1 node)"

        rnd = random.Random(seed)

        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        tree = cls()
        nodes = list(range(n))
        root = rnd.choice(nodes)
        tree.root = root

        # add root
        w_root = distribution_w()
        tree.add_node(root, c=distribution_c(), w=w_root)

        available = [(root, 0)]  # (node, number_of_children)
        sum_w = w_root

        for v in nodes:
            if v == root:
                continue
            if not available:
                raise ValueError("Cannot attach new node without exceeding degree bound Δ.")

            # choose random parent that has < delta-1 children
            parent_idx = rnd.randrange(len(available))
            parent, child_count = available[parent_idx]

            w = distribution_w()
            tree.add_node(v, c=distribution_c(), w=w)
            sum_w += w
            tree.add_edge(parent, v)

            # update parent's child count
            new_child_count = child_count + 1
            if new_child_count >= delta - 1:
                available.pop(parent_idx)
            else:
                available[parent_idx] = (parent, new_child_count)

            # add new node as potential parent
            available.append((v, 0))

        # normalize weights
        for node in tree.nodes:
            tree.nodes[node]['w'] /= sum_w

        return tree

    @classmethod
    def random_bounded_diameter(cls, n: int, D: int,
                                seed: int | None = None,
                                distribution_c=None,
                                distribution_w=None) -> Tree:
        """
        Generate a random directed rooted tree with diameter ≤ D.

        Parameters
        ----------
        n : int
            Number of nodes.
        D : int
            Maximum diameter.
        seed : int, optional
            Random seed.
        distribution_c : callable, optional
            Function returning float for node attribute 'c'.
        distribution_w : callable, optional
            Function returning float for node attribute 'w'.

        Returns
        -------
        Tree
            Randomly generated tree with diameter ≤ D.
        """
        assert n >= 1, "Tree must have at least one node"
        assert D >= 1, "Diameter bound must be at least 1"

        rnd = random.Random(seed)

        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        tree = cls()
        nodes = list(range(n))
        root = rnd.choice(nodes)
        tree.root = root

        # prepare level structure
        if D == 1:
            # only single node allowed
            tree.add_node(root, c=distribution_c(), w=1.0)
            return tree

        if D == 2:
            # star tree
            sum_w = 0
            for v in nodes:
                w = distribution_w()
                tree.add_node(v, c=distribution_c(), w=w)
                sum_w += w
            for v in nodes:
                tree.nodes[v]['w'] /= sum_w
            for v in nodes:
                if v != root:
                    tree.add_edge(root, v)
            return tree

        # general case: D ≥ 3
        h = D // 2 + 1  # number of levels (max depth)
        levels = [[] for _ in range(h)]
        remaining = n - 1
        levels[0] = [root]

        # distribute remaining nodes randomly into levels 1..h-1
        for i in range(1, h):
            # ensure each nonempty level gets at least one node until we run out
            if remaining == 0:
                break
            count = rnd.randint(1, remaining) if i < h - 1 else remaining
            available_nodes = [v for v in nodes if v not in sum(levels, [])]
            chosen = rnd.sample(available_nodes, count)
            levels[i] = chosen
            remaining -= count

        # assign attributes and build edges
        sum_w = 0
        for level in levels:
            for v in level:
                w = distribution_w()
                tree.add_node(v, c=distribution_c(), w=w)
                sum_w += w

        for i in range(1, len(levels)):
            for v in levels[i]:
                parent = rnd.choice(levels[i - 1])
                tree.add_edge(parent, v)

        # normalize weights
        for v in tree.nodes:
            tree.nodes[v]['w'] /= sum_w

        # sanity: diameter check (should be ≤ D)
        diam = nx.diameter(nx.Graph(tree))
        assert diam <= D, f"Generated diameter {diam} exceeds bound {D}"

        return tree


    @classmethod
    def random_large_degree(cls, n: int, delta: int,
                            seed: int | None = None,
                            distribution_c=None,
                            distribution_w=None) -> Tree:
        """
        Generate a random directed rooted tree with at least one vertex of degree ≥ delta.

        Parameters
        ----------
        n : int
            Number of vertices (≥ delta + 1)
        delta : int
            Desired minimum degree for at least one vertex.
        seed : int, optional
            Random seed.
        distribution_c : callable, optional
            Distribution for node attribute 'c'.
        distribution_w : callable, optional
            Distribution for node attribute 'w'.

        Returns
        -------
        Tree
            Directed tree with at least one vertex of degree ≥ delta.
        """
        assert n >= delta + 1, "n must be at least delta + 1"
        assert delta >= 2, "delta must be at least 2"

        rnd = random.Random(seed)

        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        tree = cls()
        nodes = list(range(n))
        root = rnd.choice(nodes)
        tree.root = root

        # attributes
        sum_w = 0
        for v in nodes:
            w = distribution_w()
            tree.add_node(v, c=distribution_c(), w=w)
            sum_w += w
        for v in nodes:
            tree.nodes[v]['w'] /= sum_w

        # Step 1: make root high-degree
        others = [v for v in nodes if v != root]
        high_neighbors = rnd.sample(others, delta)
        for v in high_neighbors:
            tree.add_edge(root, v)

        # Step 2: attach remaining vertices
        remaining = [v for v in others if v not in high_neighbors]
        attachable = [root] + high_neighbors
        for v in remaining:
            parent = rnd.choice(attachable)
            tree.add_edge(parent, v)
            attachable.append(v)

        # Verify property
        max_deg = max(dict(tree.degree()).values())
        assert max_deg >= delta, f"Max degree {max_deg} < {delta}"

        return tree

    @classmethod
    def random_large_diameter(cls, n: int, D: int,
                              seed: int | None = None,
                              distribution_c=None,
                              distribution_w=None) -> Tree:
        """
        Generate a random directed rooted tree with diameter ≥ D.

        Parameters
        ----------
        n : int
            Number of nodes (≥ D + 1)
        D : int
            Desired minimum diameter (≥ 1)
        seed : int, optional
            Random seed.
        distribution_c : callable, optional
            Distribution for node attribute 'c'.
        distribution_w : callable, optional
            Distribution for node attribute 'w'.

        Returns
        -------
        Tree
            Directed tree with diameter ≥ D.
        """
        assert n >= D + 1, "n must be at least D + 1"
        assert D >= 1, "D must be at least 1"

        rnd = random.Random(seed)

        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        tree = cls()
        nodes = list(range(n))

        # Build main path of length D (D+1 vertices)
        path_nodes = nodes[:D + 1]
        root = path_nodes[0]
        tree.root = root

        # add all nodes and attributes
        sum_w = 0
        for v in nodes:
            w = distribution_w()
            tree.add_node(v, c=distribution_c(), w=w)
            sum_w += w
        for v in nodes:
            tree.nodes[v]['w'] /= sum_w

        # add path edges (directed from root forward)
        for i in range(D):
            tree.add_edge(path_nodes[i], path_nodes[i + 1])

        # remaining vertices attach randomly
        remaining = nodes[D + 1:]
        for v in remaining:
            parent = rnd.choice(path_nodes)  # attach to any path node
            tree.add_edge(parent, v)

        # verify property
        diam = nx.diameter(nx.Graph(tree))
        assert diam >= D, f"Diameter {diam} < {D}"

        return tree

    @classmethod
    def all_nonisomorphic_trees(cls, n: int) -> [Tree]:
        """
        Generate all non-isomorphic trees of size n.

        Parameters
        ----------
        n : int
            Number of nodes.

        Returns
        -------
        list[Tree]
            List of non-isomorphic trees (as Tree objects).
        """
        if n < 1:
            return []
        if n == 1:
            t = cls()
            t.add_node(0)
            return [t]

        smaller_trees = cls.all_nonisomorphic_trees(n - 1)
        new_trees = []

        for tree in smaller_trees:
            for u in tree.nodes():
                new_tree = cls(tree)  # copy existing tree
                new_node = n - 1
                new_tree.add_node(new_node)
                new_tree.add_edge(u, new_node)

                # check for isomorphism
                is_new = True
                for t_existing in new_trees:
                    if nx.is_isomorphic(new_tree, t_existing):
                        is_new = False
                        break
                if is_new:
                    new_trees.append(new_tree)

        return new_trees

    @classmethod
    def enumerate_attribute_combinations(cls, tree, attributes=('c',), n=None) -> [Tree]:
        """
        For each tree in `trees`, generate all combinations of node attributes.

        Parameters
        ----------
        trees : list[Tree]
            List of non-isomorphic trees.
        attributes : tuple[str]
            Tuple of attributes to enumerate ('c' and/or 'w').
        n : int
            Number of nodes (used to scale values from 1/n to n/n).

        Returns
        -------
        dict
            Dictionary mapping original tree index -> list of trees with enumerated attributes.
        """
        assert len(attributes) > 0, "At least one attribute must be specified"

        num_nodes = len(tree.nodes) if n is None else n
        nodes_list = list(tree.nodes)
        scaled_values = [(i + 1) / num_nodes for i in range(num_nodes)]

        # Generate all product combinations for nodes
        if len(attributes) == 1:
            attr = attributes[0]
            value_combinations = product(scaled_values, repeat=num_nodes)
        else:
            # product for each node over all attributes
            value_combinations = product(
                *[product(*(scaled_values for _ in attributes)) for _ in nodes_list]
            )

        seen_ratios = set()
        tree_variants = []

        for vals in value_combinations:
            new_tree = cls(tree)
            if len(attributes) == 1:
                for node, val in zip(nodes_list, vals):
                    new_tree.nodes[node][attr] = val
                # compute ratios relative to first non-zero
                first_val = next(v for v in vals if v != 0)
                ratios = tuple(v / first_val for v in vals)
            else:
                # multiple attributes
                ratios = []
                for attr_idx, attr in enumerate(attributes):
                    attr_vals = [vals[node_idx][attr_idx] for node_idx in range(num_nodes)]
                    first_val = next(v for v in attr_vals if v != 0)
                    ratios.extend([v / first_val for v in attr_vals])
                    # set attribute values
                    for node_idx, node in enumerate(nodes_list):
                        new_tree.nodes[node][attr] = vals[node_idx][attr_idx]
                ratios = tuple(ratios)

            # unikalne proporcje
            if ratios not in seen_ratios:
                seen_ratios.add(ratios)
                tree_variants.append(new_tree)

        return tree_variants

    @classmethod
    def random_spider(cls, n: int, legs: int = None, seed: int | None = None,
                      distribution_c=None, distribution_w=None) -> Tree:
        """
        Generate a random spider tree with optional number of legs.

        Parameters
        ----------
        n : int
            Total number of vertices
        legs : int, optional
            Number of legs (if None, chosen randomly between 1 and n-1)
        seed : int, optional
            Random seed
        distribution_c : callable, optional
            Node attribute 'c'
        distribution_w : callable, optional
            Node attribute 'w'

        Returns
        -------
        Tree
            Random spider tree
        """
        assert n >= 1, "Tree must have at least one node"
        rnd = random.Random(seed)
        tree = cls()

        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        # Root
        tree.add_node(0, c=distribution_c(), w=0)
        tree.root = 0

        if n == 1:
            tree.nodes[0]['w'] = 1.0
            return tree

        # Określenie liczby nóg
        if legs is None:
            legs = rnd.randint(1, n - 1)
        else:
            assert 1 <= legs <= n - 1, "legs must be between 1 and n-1"

        remaining_nodes = n - 1
        leg_lengths = [1] * legs
        remaining_nodes -= legs

        # Rozdziel pozostałe węzły losowo po nogach
        for _ in range(remaining_nodes):
            idx = rnd.randint(0, legs - 1)
            leg_lengths[idx] += 1

        # Budowanie nóg
        node_id = 1
        for length in leg_lengths:
            prev = 0  # start from root
            for _ in range(length):
                tree.add_node(node_id, c=distribution_c(), w=distribution_w())
                tree.add_edge(prev, node_id)
                prev = node_id
                node_id += 1

        # Normalizacja wag
        sum_w = sum(tree.nodes[v]['w'] for v in tree.nodes)
        for v in tree.nodes:
            tree.nodes[v]['w'] /= sum_w

        return tree

    @classmethod
    def random_caterpillar(cls, n: int, spine_length: int = None, seed: int | None = None,
                           distribution_c=None, distribution_w=None) -> 'Tree':
        """
        Generate a random caterpillar tree.

        Parameters
        ----------
        n : int
            Total number of vertices.
        spine_length : int, optional
            Length of the central path (number of vertices on the spine).
            If None, chosen randomly between 2 and n.
        seed : int, optional
            Random seed.
        distribution_c : callable, optional
            Node attribute 'c'.
        distribution_w : callable, optional
            Node attribute 'w'.

        Returns
        -------
        Tree
            Random caterpillar tree.
        """
        assert n >= 2, "Caterpillar must have at least 2 vertices"
        rnd = random.Random(seed)
        tree = cls()

        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        # Losowa długość kręgosłupa
        if spine_length is None:
            spine_length = rnd.randint(2, n)

        # Utwórz kręgosłup
        for i in range(spine_length):
            tree.add_node(i, c=distribution_c(), w=distribution_w())
            if i > 0:
                tree.add_edge(i - 1, i)
        tree.root = 0

        # Pozostałe wierzchołki to liście
        remaining = n - spine_length
        next_id = spine_length

        for _ in range(remaining):
            parent = rnd.randint(0, spine_length - 1)
            tree.add_node(next_id, c=distribution_c(), w=distribution_w())
            tree.add_edge(parent, next_id)
            next_id += 1

        # Normalizacja wag
        sum_w = sum(tree.nodes[v]['w'] for v in tree.nodes)
        for v in tree.nodes:
            tree.nodes[v]['w'] /= sum_w

        return tree

    @classmethod
    def random_star(cls, n: int, seed: int | None = None,
             distribution_c=None, distribution_w=None) -> 'Tree':
        """Generate a star tree with one central node."""
        rnd = random.Random(seed)
        tree = cls()
        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        # Root = centrum
        tree.add_node(0, c=distribution_c(), w=distribution_w())
        tree.root = 0

        for i in range(1, n):
            tree.add_node(i, c=distribution_c(), w=distribution_w())
            tree.add_edge(0, i)

        cls._normalize_w(tree)
        return tree

    @classmethod
    def random_full_mary(cls, n: int, m: int, seed: int | None = None,
                  distribution_c=None, distribution_w=None) -> 'Tree':
        """Generate a full m-ary tree (each node has 0 or m children)."""
        rnd = random.Random(seed)
        tree = cls()
        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        tree.add_node(0, c=distribution_c(), w=distribution_w())
        tree.root = 0

        queue = [0]
        node_id = 1

        while queue and node_id < n:
            parent = queue.pop(0)
            for _ in range(m):
                if node_id >= n:
                    break
                tree.add_node(node_id, c=distribution_c(), w=distribution_w())
                tree.add_edge(parent, node_id)
                queue.append(node_id)
                node_id += 1

        cls._normalize_w(tree)
        return tree

    @classmethod
    def random_balanced(cls, n: int, m: int = 2, seed: int | None = None,
                 distribution_c=None, distribution_w=None) -> 'Tree':
        """Generate a balanced m-ary tree."""
        rnd = random.Random(seed)
        tree = cls()
        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        tree.add_node(0, c=distribution_c(), w=distribution_w())
        tree.root = 0
        node_id = 1
        current_level = [0]

        while node_id < n:
            next_level = []
            for parent in current_level:
                for _ in range(m):
                    if node_id >= n:
                        break
                    tree.add_node(node_id, c=distribution_c(), w=distribution_w())
                    tree.add_edge(parent, node_id)
                    next_level.append(node_id)
                    node_id += 1
            current_level = next_level

        cls._normalize_w(tree)
        return tree

    @classmethod
    def random_broom(cls, n: int, handle_length: int = None, seed: int | None = None,
              distribution_c=None, distribution_w=None) -> 'Tree':
        """Generate a broom tree: a path (handle) with a star at the end."""
        rnd = random.Random(seed)
        tree = cls()
        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        if handle_length is None:
            handle_length = rnd.randint(1, n - 1)
        handle_length = min(handle_length, n - 1)

        # Uchwyt (ścieżka)
        for i in range(handle_length):
            tree.add_node(i, c=distribution_c(), w=distribution_w())
            if i > 0:
                tree.add_edge(i - 1, i)

        # Gwiazda na końcu uchwytu
        root_star = handle_length - 1
        remaining = n - handle_length
        next_id = handle_length
        for _ in range(remaining):
            tree.add_node(next_id, c=distribution_c(), w=distribution_w())
            tree.add_edge(root_star, next_id)
            next_id += 1

        tree.root = 0
        cls._normalize_w(tree)
        return tree

    @classmethod
    def random_lobster(cls, n: int, spine_length: int = None, seed: int | None = None,
                distribution_c=None, distribution_w=None) -> 'Tree':
        """Generate a lobster tree (caterpillar with leaves-of-leaves)."""
        rnd = random.Random(seed)
        tree = cls()
        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        if spine_length is None:
            spine_length = rnd.randint(2, n // 2)

        # Tworzymy kręgosłup
        for i in range(spine_length):
            tree.add_node(i, c=distribution_c(), w=distribution_w())
            if i > 0:
                tree.add_edge(i - 1, i)

        remaining = n - spine_length
        next_id = spine_length
        spine_nodes = list(range(spine_length))

        # Etap 1: liście do kręgosłupa
        leaf_nodes = []
        while remaining > 0:
            parent = rnd.choice(spine_nodes)
            tree.add_node(next_id, c=distribution_c(), w=distribution_w())
            tree.add_edge(parent, next_id)
            leaf_nodes.append(next_id)
            next_id += 1
            remaining -= 1
            if remaining <= 0:
                break

            # Etap 2: część liści dostaje swoje liście
            if rnd.random() < 0.5 and remaining > 0:
                tree.add_node(next_id, c=distribution_c(), w=distribution_w())
                tree.add_edge(leaf_nodes[-1], next_id)
                next_id += 1
                remaining -= 1

        tree.root = 0
        cls._normalize_w(tree)
        return tree

    @classmethod
    def random_probabilistic(cls, n: int, offspring_distribution=None, seed: int | None = None,
                      distribution_c=None, distribution_w=None) -> 'Tree':
        """
        Generate a random tree using a Galton–Watson branching process.

        Parameters
        ----------
        n : int
            Number of vertices.
        offspring_distribution : callable -> int
            Function returning the number of children of a node.
        seed : int
            Random seed.
        distribution_c, distribution_w : callable
            Functions generating node attributes.
        """
        rnd = random.Random(seed)
        if offspring_distribution is None:
            # Domyślnie Poisson(λ=1.5)
            offspring_distribution = lambda: max(0, rnd.poisson(1.5)) if hasattr(rnd, 'poisson') else rnd.randint(0, 3)

        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        tree = cls()
        tree.add_node(0, c=distribution_c(), w=distribution_w())
        tree.root = 0

        queue = [0]
        node_id = 1

        while queue and node_id < n:
            parent = queue.pop(0)
            k = offspring_distribution()
            for _ in range(k):
                if node_id >= n:
                    break
                tree.add_node(node_id, c=distribution_c(), w=distribution_w())
                tree.add_edge(parent, node_id)
                queue.append(node_id)
                node_id += 1

        # Normalizacja wag
        total_w = sum(tree.nodes[v]['w'] for v in tree.nodes)
        for v in tree.nodes:
            tree.nodes[v]['w'] /= total_w

        return tree

    @classmethod
    def random_recursive(cls, n: int, seed: int | None = None,
                  distribution_c=None, distribution_w=None) -> 'Tree':
        """
        Generate a random recursive tree.

        Model: each new node attaches to a uniformly random existing node.

        Parameters
        ----------
        n : int
            Number of vertices.
        seed : int
            Random seed.
        """
        rnd = random.Random(seed)
        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        tree = cls()
        tree.add_node(0, c=distribution_c(), w=distribution_w())
        tree.root = 0

        for i in range(1, n):
            parent = rnd.randint(0, i - 1)
            tree.add_node(i, c=distribution_c(), w=distribution_w())
            tree.add_edge(parent, i)

        # Normalizacja wag
        total_w = sum(tree.nodes[v]['w'] for v in tree.nodes)
        for v in tree.nodes:
            tree.nodes[v]['w'] /= total_w

        return tree

    @classmethod
    def random_preferential(cls, n: int, seed: int | None = None,
                     distribution_c=None, distribution_w=None) -> 'Tree':
        """
        Generate a preferential-attachment random tree (Barabási–Albert type).
        """
        rnd = random.Random(seed)
        if distribution_c is None:
            distribution_c = lambda: rnd.uniform(0, 1)
        if distribution_w is None:
            distribution_w = lambda: rnd.uniform(0, 1)

        tree = cls()
        tree.add_node(0, c=distribution_c(), w=distribution_w())
        tree.root = 0

        degrees = {0: 1}  # zaczynamy od stopnia 1, żeby uniknąć zera

        for i in range(1, n):
            # losujemy wierzchołek z prawdopodobieństwem proporcjonalnym do stopnia
            nodes, weights = zip(*degrees.items())
            total = sum(weights)
            r = rnd.uniform(0, total)
            acc = 0
            for node, w in zip(nodes, weights):
                acc += w
                if r <= acc:
                    parent = node
                    break
            tree.add_node(i, c=distribution_c(), w=distribution_w())
            tree.add_edge(parent, i)
            degrees[parent] += 1
            degrees[i] = 1

        # Normalizacja wag
        total_w = sum(tree.nodes[v]['w'] for v in tree.nodes)
        for v in tree.nodes:
            tree.nodes[v]['w'] /= total_w

        return tree

    def make_monotone_increasing(self, attr='c', rnd=None):
        """
        Przekształca funkcję tak, aby była rosnąca od jakiegoś wierzchołka.
        Jeśli wierzchołek ma wartość mniejszą niż rodzic, losuje nową wartość
        z przedziału [rodzic, 1].
        """
        if rnd is None:
            rnd = random.Random()

        def dfs(v):
            for u in self.successors(v):
                min_val = self.nodes[v][attr]
                if self.nodes[u][attr] < min_val:
                    # losujemy nową wartość z przedziału [min_val, 1]
                    self.nodes[u][attr] = rnd.uniform(min_val, 1)
                dfs(u)

        # root = wierzchołek z minimalną wartością
        new_root = min(self.nodes, key=lambda x: self.nodes[x][attr])
        self.root = new_root
        dfs(new_root)

    def make_monotone_decreasing(self, attr='c', rnd=None):
        """
        Przekształca funkcję tak, aby była malejąca od jakiegoś wierzchołka.
        Jeśli wierzchołek ma wartość większą niż rodzic, losuje nową wartość
        z przedziału [0, rodzic].
        """
        if rnd is None:
            rnd = random.Random()

        def dfs(v):
            for u in self.successors(v):
                max_val = self.nodes[v][attr]
                if self.nodes[u][attr] > max_val:
                    # losujemy nową wartość z przedziału [0, max_val]
                    self.nodes[u][attr] = rnd.uniform(0, max_val)
                dfs(u)

        # root = wierzchołek z maksymalną wartością
        new_root = max(self.nodes, key=lambda x: self.nodes[x][attr])
        self.root = new_root
        dfs(new_root)

    def make_k_up_modular(self, k: int, attr='c'):
        """
        Przekształca wartości `attr` w k-up-modularną funkcję.
        """
        # Sortowanie wierzchołków malejąco według wartości
        nodes_sorted = sorted(self.nodes(data=True), key=lambda x: -x[1].get(attr, 0))

        for node, data in nodes_sorted:
            s = data[attr]
            heavy_groups = self.get_heavy_groups(s)
            while len(heavy_groups) > k:
                h1 = heavy_groups[0]



                h2 = heavy_groups[1]
                hv1 = list(h1.nodes())[0]
                hv2 = list(h2.nodes())[0]
                path = self.minimal_subtree({hv1, hv2})
                for v in path.nodes:
                    self.nodes[v][attr] = s
                heavy_groups = self.get_heavy_groups(s)
