from __future__ import annotations

from tree import Tree


class DecisionTree(Tree):

    def cost(self):
        r = self.find_root()
        if len(self) == 1:
            return self.weight(r)
        tree_copy = self.copy()
        return self.weight(r) + max([t.cost() for t in tree_copy.ccs(r)])

    def __lt__(self, other):
        return self.cost() < other.cost()

    def query(self, tree) -> dict[Tree: DecisionTree]:
        r = self.find_root()
        t_ccs = tree.ccs(r)
        dt_ccs = self.ccs(r)
        responses = {}
        for t in t_ccs:
            for dt in dt_ccs:
                if set(dt.nodes).issubset(t.nodes):
                    responses[t] = dt
        return responses

    def find_last_q_consistent_w_subtree(self, tree, subtree: Tree):
        for t, dt in self.query(tree).items():
            if subtree.is_subtree(t):
                return dt.find_last_q_consistent_w_subtree(t, subtree)
        return self.find_root()

    def attach_sub_dt(self, tree, h, dt_h):
        q = self.find_last_q_consistent_w_subtree(tree, h)
        self.attach_subtree(dt_h, q)
