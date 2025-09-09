from __future__ import annotations

import copy

from tree import Tree


class DecisionTree(Tree):

    def cost(self):
        r = self.get_root()
        if len(self) == 1:
            return self.vcost(r)
        tree_copy = self.copy()
        cost = self.vcost(r) + max([DecisionTree(t).cost() for t in tree_copy.ccs(r)])
        return cost

    def __lt__(self, other: Tree):
        return self.cost() < other.cost()

    def query(self, tree: Tree, query: int = None, excluded_responses=None) -> dict[Tree: DecisionTree]:
        if query is None:
            query = self.get_root()
        t_ccs = tree.ccs(query)
        responses = {}
        for t in t_ccs:
            if excluded_responses is None or all(v not in excluded_responses for v in t.nodes()):
                nodes_to_delete = tree.nodes() - t.nodes()
                dt = copy.deepcopy(DecisionTree(self))
                dt.remove_nodes_from(nodes_to_delete)
                responses[t] = DecisionTree(dt)
        return responses

    def find_last_q_consistent_w_subtree(self, tree, subtree: Tree):
        for t, dt in self.query(tree).items():
            if t is not None:
                if subtree.nodes == t.nodes:
                    return self.get_root()
                if subtree.is_subtree(t):
                    return dt.find_last_q_consistent_w_subtree(t, subtree)
        return self.get_root()

    def attach_sub_dt(self, tree, h, dt_h):
        q = self.find_last_q_consistent_w_subtree(tree, h)
        self.attach_subtree(dt_h, q)

    def replace_excluded_queries(self, tree, excluded_queries) -> DecisionTree:
        root = self.get_root()
        tree = copy.deepcopy(tree)
        if root in excluded_queries:
            new_dt = None
            last_v = None
            for v in tree.get_neighbors(root):
                if new_dt is None:
                    new_dt = DecisionTree(v)
                else:
                    new_dt.add_node(v)
                    new_dt.add_edge(last_v, v)
                last_v = v
                for response_tree, response_dt in self.query(tree, v):
                    if root in response_tree:
                        tree = response_tree
                    else:
                        new_dt_response = response_dt.replace_excluded_queries(tree=response_tree,
                                                                               excluded_queries=excluded_queries)
                        new_dt.attach_subtree(new_dt_response, v)
        else:
            new_dt = DecisionTree(root)
            for cc, dt_cc in self.query(tree):
                dt_cc = DecisionTree(dt_cc)
                new_dt_cc = dt_cc.replace_excluded_queries(tree=cc, excluded_queries=excluded_queries)
                new_dt.attach_subtree(new_dt_cc, root)
        return new_dt
