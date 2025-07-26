from __future__ import annotations

from math import floor

from decision_tree import DecisionTree
import copy
from itertools import product


class PartialDT(DecisionTree):
    def __init__(self, base: any = None, box_size=1, max_depth=1, slot_size=0.1):
        super().__init__()
        if isinstance(base, PartialDT):
            self.box_size = base.box_size
            self.max_depth = base.max_depth
            self.slot_size = base.slot_size
            # Deep copy nodes with their attributes
            for node, data in base.nodes(data=True):
                self.add_node(node, **copy.deepcopy(data))

            # Copy edges
            for u, v, edge_data in base.edges(data=True):
                self.add_edge(u, v, **copy.deepcopy(edge_data))
        else:
            self.box_size = box_size
            self.max_depth = max_depth
            self.slot_size = slot_size
            for i in range(max_depth):
                self.add_node(i, qs=[], load=base[i] if isinstance(base, list) else 0.0)
                if i > 0:
                    self.add_edge(i, i - 1)
        return

    def remove_query(self, query):
        root = self.get_root()
        queries_with_right_sub_dts = list(self.nodes(data=True)[root]['qs'])
        root_query_with_sub_dts = queries_with_right_sub_dts[0]
        root_query = root_query_with_sub_dts[0]
        if query == root_query:
            queries_with_right_sub_dts.pop(0)
            self.nodes(data=True)[root]['qs'] = queries_with_right_sub_dts
            if len(queries_with_right_sub_dts) == 0:
                self.remove_node(root[0])
                if len(self) > 0:
                    self.remove_query(query=query)

    def to_decision_tree(self) -> DecisionTree:
        root = self.get_root()
        queries_with_right_sub_dts = list(self.nodes(data=True)[root]['qs'])
        root_query_with_sub_dts = queries_with_right_sub_dts[0]
        root_query = root_query_with_sub_dts[0]
        weight = root_query_with_sub_dts[1]['w']
        sub_dts = root_query_with_sub_dts[1]["right_dts"]
        root_dt = {root_query: {'w': weight}}
        dt = DecisionTree(root_dt)
        self.remove_query(root_query)
        if len(self) > 0:
            sub_dts.append(self)
        for sub_dt in sub_dts:
            sub_dt = PartialDT(sub_dt)
            sub_dt_dt = sub_dt.to_decision_tree()
            dt.attach_subtree(sub_dt_dt, root_dt)
        return dt

    def cost(self):
        root = self.get_root()
        queries_with_right_sub_dts = list(self.nodes(data=True)[root]['qs'])
        root_query_with_sub_dts = queries_with_right_sub_dts[0]
        sub_dts = root_query_with_sub_dts["right_dts"]
        return max(self.box_size * self.max_depth, self.box_size + max([sub_dt.cost() for sub_dt in sub_dts]))

    def put_query(self, box_index, query, weight, sub_dts=None):
        if sub_dts is None:
            sub_dts = []
        box = self.nodes(data=True)[box_index][1]
        if box['load'] + weight > self.box_size:
            raise Exception
        else:
            new_qs = list(box['qs'])
            weight = weight - self.box_size
            if weight <= 0:
                new_qs.append({query: {'right_dts': sub_dts}})
            else:
                new_qs.append({query: {'right_dts': []}})
                self.put_query(box_index + 1, query, weight, sub_dts)
            box = {'load': min(self.box_size, weight), 'qs': new_qs}
            self.nodes(data=True)[box_index][1] = box

    def all_bipartitions(self) -> tuple[list[float], list[float]]:
        options = [[]] * self.max_depth
        for i in range(self.max_depth):
            node = self.nodes(data=True)
            load = node['load']
            empty_slots = floor(load / self.slot_size)
            for slot_amount in range(0, empty_slots):
                options[i].append(i * self.slot_size)
        combinations = list(product(*options))
        for combination in combinations:
            combination2 = [self.nodes(data=True)[i][1]["load"] - combination[i] for i in range(self.max_depth)]
            yield combination, combination2

    def get_loads(self) -> list[float]:
        return [node['load'] for node in self.nodes(data=True)]

    def merge(self, other: PartialDT, box: int, query):
        for i in range(box + 1):
            self.nodes[i]['load'] = self.nodes[i]['load'] + other.nodes[i]['load']
            self.nodes[i]['qs'] = self.nodes[i]['qs'] + other.nodes[i]['qs']
        rotated_subtree = other.get_subtree(box)
        self.nodes[box]['qs'][query].append(rotated_subtree)

    def query_sequence(self, box):
        return [node[0] for node in self.nodes(data=True)[box][1]['qs']]
