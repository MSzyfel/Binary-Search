from __future__ import annotations


class Separator:
    def __init__(self, vertices=None):
        self.vset = {}
        self.cost = 0
        if vertices is None:
            return
        if not isinstance(vertices, list):
            vertices = [vertices]
        for vset in vertices:
            if isinstance(vset, dict):
                self.vset |= vset
                self.cost += list(vset.values())[0]['c']
            elif isinstance(vset, Separator):
                self.vset |= vset.vset
                self.cost += vset.cost

    def add(self, other: Separator):
        self.vset.extend(other.vset)
        self.cost += other.cost
