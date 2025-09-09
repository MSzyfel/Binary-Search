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
                self.add_node(i, qs={}, load=base[i][0] if isinstance(base, list) else 0.0,
                              trans=base[i][1] if isinstance(base, list) else False, first=None)
                if i > 0:
                    self.add_edge(i - 1, i)
        return

    def remove_query(self, query):
        root = self.get_root()
        self.nodes[root]['first'] = None
        queries_with_right_right_dts = list(self.nodes(data=True)[root]['qs'].items())
        for root_query_with_right_dts in queries_with_right_right_dts:
            root_query = root_query_with_right_dts[0]
            if query == root_query:
                queries_with_right_right_dts.remove(root_query_with_right_dts)
                self.nodes[root]['qs'] = queries_with_right_right_dts
                if len(queries_with_right_right_dts) == 0:
                    self.remove_node(root)
                    if len(self) > 0:
                        self.remove_query(query=query)

    def to_decision_tree(self) -> DecisionTree:
        root = self.get_root()
        queries_with_right_right_dts = list(self.nodes(data=True)[root]['qs'].items())
        if len(queries_with_right_right_dts) == 0:
            self.remove_node(root)
            if len(self):
                return self.to_decision_tree()
            else:
                return DecisionTree()
        if self.nodes[root]['first'] is not None:
            root_query_with_right_dts = self.nodes(data=True)[root]['first']
        else:
            root_query_with_right_dts = queries_with_right_right_dts[0]
        root_query = root_query_with_right_dts[0]
        cost = root_query_with_right_dts[1]['c']
        right_dts = root_query_with_right_dts[1]["right_dts"]
        root_dt = {root_query: {'c': cost}}
        dt = DecisionTree(root_dt)
        self.remove_query(root_query)
        if len(self) > 0:
            right_dts.append(self)
        for sub_dt in right_dts:
            sub_dt = PartialDT(sub_dt)
            sub_dt_dt = sub_dt.to_decision_tree()
            dt.attach_subtree(sub_dt_dt, root_dt)
        return dt

    def cost(self):
        box = self.get_root()
        queries_with_right_right_dts = list(self.nodes(data=True)[box]['qs'].items())
        sub_dts = []
        for root_query_with_right_dts in queries_with_right_right_dts:
            sub_dts.extend(root_query_with_right_dts[1]["right_dts"])
        self_copy = copy.deepcopy(self)
        self_copy.remove_node(box)
        if box < self.max_depth - 1:
            sub_dts.append(self_copy)
        # TODO: check if this cost function is correct
        if len(sub_dts) == 0:
            return self.box_size
        return self.box_size + max(sub_dt.cost() for sub_dt in sub_dts)

    def put_query(self, box_index, slot_index, query, cost, is_heavy: bool, is_first=False, right_dts=None):
        if right_dts is None:
            right_dts = []
        box = self.nodes(data=True)[box_index]
        new_qs = box['qs']
        load = box['load']
        contribution = min(cost, self.box_size - slot_index * self.slot_size)
        new_cost = cost - contribution
        if (is_heavy and box['load'] > 0) or box['load'] == self.box_size or box['load'] + contribution > self.box_size:
            raise Exception
        else:
            if new_cost == 0:
                new_qs[query] = {'c': cost, 'right_dts': right_dts}
                box = {'load': load + contribution, 'qs': new_qs, 'trans': False,
                       'first': query if is_first else None}
            else:
                if box['trans'] is True:
                    raise Exception
                new_qs[query] = {'c': cost, 'right_dts': []}
                self.put_query(box_index + 1, 0, query, new_cost, is_heavy, True, right_dts)
                box = {'load': box['load'] + contribution, 'qs': new_qs, 'trans': True,
                       'first': query if is_first else None}
            self.nodes[box_index].update(box)

    def all_bipartitions(self) -> tuple[list[float], list[float]]:
        options = []
        nodes = self.nodes(data=True)
        for i in range(self.max_depth):
            options_for_box = []
            node = nodes[i]
            load = node['load']
            empty_slots = int(round((self.box_size - load) / self.slot_size, 0))
            for j in range(0, empty_slots + 1):
                if node['trans']:
                    options_for_box.append((j * self.slot_size, True))
                else:
                    options_for_box.append((j * self.slot_size, True))
                    options_for_box.append((j * self.slot_size, False))
            options.append(options_for_box)
        combinations = list(product(*options))
        for combination in combinations:
            combination2 = [
                (self.box_size - self.nodes[i]['load'] - combination[i][0], not combination[i][1] or nodes[i]['trans'])
                for i in range(self.max_depth)]
            yield combination, combination2

    def get_loads(self) -> list[tuple[float, bool]]:
        return [(node[1]['load'], node[1]['trans']) for node in self.nodes(data=True)]

    def merge(self, other: PartialDT, box: int, query):
        for i in range(box + 1):
            self.nodes[i]['load'] = self.nodes[i]['load'] + other.nodes[i]['load']
            self.nodes[i]['qs'] = self.nodes[i]['qs'] | other.nodes[i]['qs']
            self.nodes[i]['trans'] = self.nodes[i]['trans'] or other.nodes[i]['trans']
            self.nodes[i]['trans'] = self.nodes[i]['first'] if self.nodes[i]['first'] is not None else other.nodes[i][
                'first']
        rotated_subtree = other.get_subtree(box)
        self.nodes[box]['qs'][query].append(rotated_subtree)

    def query_sequence(self, box):
        return [node for node in self.nodes[box]['qs']]
