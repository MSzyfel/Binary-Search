from __future__ import annotations

from tree import Tree


class DecisionTree(Tree):

    def cost(self):
        r = self.get_root()
        if len(self) == 1:
            return self.weight(r)
        tree_copy = self.copy()
        cost = self.weight(r) + max([DecisionTree(t).cost() for t in tree_copy.ccs(r)])
        return cost

    def __lt__(self, other: Tree):
        return self.cost() < other.cost()

    def query(self, tree: Tree, excluded_responses=None) -> dict[Tree: DecisionTree]:
        query = self.get_root()
        t_ccs = tree.ccs(query)
        dt_ccs = self.ccs(query)
        responses = {}
        for t in t_ccs:
            if excluded_responses is None or all(v not in excluded_responses for v in t.nodes()):
                for dt in dt_ccs:
                    if set(dt.nodes).issubset(t.nodes):
                        responses[t] = DecisionTree(dt)
        return responses

    def find_last_q_consistent_w_subtree(self, tree, subtree: Tree):
        for t, dt in self.query(tree).items():
            if t is not None and subtree.is_subtree(t):
                return dt.find_last_q_consistent_w_subtree(t, subtree)
        return self.get_root()

    def attach_sub_dt(self, tree, h, dt_h):
        q = self.find_last_q_consistent_w_subtree(tree, h)
        self.attach_subtree(dt_h, q)
