from __future__ import annotations

from tree import Tree


class DecisionTree(Tree):
    def cost(self):
        r = self.find_root()
        if len(self) == 1:
            return r.get('c')
        tree_copy = self.copy()
        return r.get('c') + max([t.cost() for t in tree_copy.ccs(r)])

    def __lt__(self, other):
        return self.cost() < other.cost()

