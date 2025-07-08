from __future__ import annotations

from itertools import combinations

import networkx as nx


class Tree(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, attr)
        self.r = None

    def __hash__(self):
        nodes = sorted(self.nodes)
        g_hash = 0
        node_iterator = 0
        for i in range(0, nodes[len(nodes) - 1]):
            if nodes[node_iterator] == i:
                g_hash = g_hash + 2 ** i
                node_iterator = node_iterator + 1
        return g_hash

    def weight(self, v):
        return self.nodes[v]['w']

    def find_root(self):
        if self.r is None:
            r = next(v for v in self.nodes if self.nodes[v].in_degree == 0)
            self.r = r
        else:
            return self.r

    def ccs(self, vertices) -> list[Tree]:
        G = self.copy()
        if not isinstance(vertices, set):
            vertices = set(vertices)
        G.remove_nodes_from(vertices)
        undir_G_minus_v = G.to_undirected()

        subtrees = []
        for component in nx.connected_components(undir_G_minus_v):
            subgraph = self.subgraph(component).copy()
            subtree = Tree()
            subtree.add_nodes_from(subgraph.nodes(data=True))
            subtree.add_edges_from(subgraph.edges(data=True))
            subtrees.append(subtree)

        return subtrees

    def attach_subtree(self, other: Tree, v):
        root_other = other.find_root()
        self.add_nodes_from(other.nodes(data=True))
        self.add_edges_from(other.edges(data=True))
        self.add_edge(v, root_other)

    def centroid(self):
        v = self.find_root()
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
        for u in subtree.nodes:
            if self.has_edge(u, v):
                return u
        return None

    def is_descendant(self, ancestor, node) -> bool:
        return nx.has_path(self, source=ancestor, target=node)

    def pairs_of_ordered_vertices(self, nodes: set):
        for v in nodes:
            for u in nodes:
                if not self.is_descendant(v, u):
                    yield u, v

    def minimal_subtree(self, terminals: set) -> Tree:
        nodes = set()
        edges = set()

        for u, v in self.pairs_of_ordered_vertices(terminals):
            path = nx.shortest_path(self, source=v, target=u)
            nodes.update(path)
            edges.update(zip(path[:-1], path[1:]))

        return self.edge_subgraph(edges).copy()

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
        subtree = Tree()
        subtree.add_nodes_from(terminals)
        for u, v in self.pairs_of_ordered_vertices(terminals):
            path_u_v = nx.shortest_path(self, source=v, target=u).nodes
            set_of_vertices_of_path_u_v = set(path_u_v)
            if terminals & set_of_vertices_of_path_u_v == {u, v}:
                if len(set_of_vertices_of_path_u_v) == 2:
                    subtree.add_edge(u, v)
                else:
                    l = min([set_of_vertices_of_path_u_v - {u, v}], key=lambda v: self.weight(v))
                    subtree.add_nodes_from(l)
                    subtree.add_edge(u, l)
                    subtree.add_edge(l, v)
        return subtree

    def is_subtree(self, subtree: Tree) -> bool:
        return set(self.nodes).issubset(subtree.nodes) and set(self.edges).issubset(subtree.edges)
