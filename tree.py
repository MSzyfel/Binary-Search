from __future__ import annotations

import networkx as nx


class Tree(nx.DiGraph):
    def __hash__(self):
        nodes = sorted(self.nodes)
        g_hash = 0
        node_iterator = 0
        for i in range(0, nodes[len(nodes) - 1]):
            if nodes[node_iterator] == i:
                g_hash = g_hash + 2 ** i
                node_iterator = node_iterator + 1
        return g_hash

    def find_root(self):
        return next(v for v in self.nodes if self.nodes[v].in_degree == 0)

    def ccs(self, v) -> list[Tree]:
        G = self.copy()
        G.remove_node(v)
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

    def minimal_subtree(self, terminals: set) -> Tree:
        nodes = set()
        edges = set()

        for v in terminals:
            for u in terminals:
                if self.is_descendant(v, u):
                    continue
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
