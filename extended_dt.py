from __future__ import annotations

from math import floor

from decision_tree import DecisionTree
import copy
from itertools import product

import networkx as nx


class ExtendedDT(DecisionTree):
    def __init__(self, base: any = None, box_size=1, max_depth=0, slot_size=1):
        super().__init__()
        if isinstance(base, ExtendedDT):
            self.box_size = base.box_size
            self.max_depth = base.max_depth
            self.slot_size = base.slot_size
            # Deep copy nodes with their attributes
            for node, data in base.nodes(data=True):
                self.add_node(node, **copy.deepcopy(data))

            # Copy edges
            for u, v, edge_data in base.edges(data=True):
                self.add_edge(u, v, **copy.deepcopy(edge_data))
        # if isinstance(base, )
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

    def get_subtree(self, v):
        T = ExtendedDT()
        T.box_size = self.box_size
        T.max_depth = self.max_depth
        T.slot_size = self.slot_size
        T.graph.update(copy.deepcopy(self.graph))

        def dfs(node):
            if node in T.nodes:
                return
            # Dodaj węzeł z atrybutami
            T.add_node(node, **copy.deepcopy(self.nodes[node]))
            for child in self.successors(node):
                dfs(child)
                # Dodaj krawędź dopiero po odwiedzeniu dziecka
                T.add_edge(node, child, **copy.deepcopy(self.edges[node, child]))

        dfs(v)
        return T

    def remove_query(self, query):
        for box in self.nodes:
            # self.nodes[root]['first'] = None
            if self.nodes[box]['first'] == query:
                self.nodes[box]['first'] = None
            queries_with_right_right_dts = list(self.nodes(data=True)[box]['qs'].items())
            for root_query_with_right_dts in queries_with_right_right_dts:
                root_query = root_query_with_right_dts[0]
                if query == root_query:
                    queries_with_right_right_dts.remove(root_query_with_right_dts)
                    self.nodes[box]['qs'] = {key: value for key, value in queries_with_right_right_dts}
                    # if len(queries_with_right_right_dts) == 0:
                    #     self.remove_node(box)
                    #     if len(self) > 0:
                    #         self.remove_query(query=query)

    def to_decision_tree(self) -> DecisionTree:
        root = self.get_root()
        print("processed subtree")
        self.print_tree()
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
            root_query_with_right_dts = min(
                queries_with_right_right_dts,
                key=lambda x: x[1]['c']
            )
        root_query = root_query_with_right_dts[0]
        cost = root_query_with_right_dts[1]['c']
        right_dts = []
        for box in range(root, self.max_depth):
            if root_query in self.nodes[box]['qs']:
                right_dts.extend(self.nodes[box]['qs'][root_query]['right_dts'])
        dt = DecisionTree(root_query)
        dt.nodes[root_query]['c']= cost
        self.print_tree()
        self.remove_query(root_query)
        self.print_tree()
        if len(self) > 0:
            right_dts.insert(0, self)
        for sub_dt in right_dts:
            sub_dt = ExtendedDT(sub_dt)
            sub_dt_dt = sub_dt.to_decision_tree()
            dt.attach_subtree(sub_dt_dt, dt.get_root())
        print(dt.nodes(data=True))
        return dt

    def cost(self, crit='worst'):
        box = self.get_root()
        queries_with_right_right_dts = list(self.nodes(data=True)[box]['qs'].items())
        sub_dts = []
        for root_query_with_right_dts in queries_with_right_right_dts:
            sub_dts.extend(root_query_with_right_dts[1]["right_dts"])
        self_copy = copy.deepcopy(self)
        self_copy.remove_node(box)
        if crit == 'worst':
            cost_left = self_copy.cost() if box < self.max_depth - 1 else 0
            return max(cost_left+self.box_size, max(sub_dt.cost(crit) for sub_dt in sub_dts) if len(sub_dts) > 0 else 0)
        # TODO: check if this cost function is correct
        else:
            if box < self.max_depth - 1:
                sub_dts.append(self_copy)
            return self.box_size * self.sum_of('w') + (
                sum(sub_dt.cost(crit) for sub_dt in sub_dts) if len(sub_dts) > 0 else 0)

    def put_query(self, box_index, slot_index, query, cost, is_heavy: bool, is_first=False, weight = 1):
        box = self.nodes(data=True)[box_index]
        new_qs = box['qs']
        load = box['load']
        contribution = min(cost, self.box_size - slot_index * self.slot_size)
        new_cost = cost - contribution
        if (is_heavy and box['load'] > 0) or box['load'] == self.box_size or box['load'] + contribution > self.box_size:
            raise Exception
        else:
            if new_cost == 0:
                new_qs[query] = {'c': cost, 'right_dts': [], 'w': weight}
                box = {'load': load + contribution, 'qs': new_qs, 'trans': False,
                       'first': query if is_first else None}
            else:
                if box['trans'] is True:
                    raise Exception
                new_qs[query] = {'c': cost, 'right_dts': [], 'w': weight}
                self.put_query(box_index + 1, 0, query, new_cost, is_heavy, True)
                box = {'load': box['load'] + contribution, 'qs': new_qs, 'trans': True,
                       'first': query if is_first else None}
            self.nodes[box_index].update(box)

    def all_bipartitions(self, trans=True) -> tuple[list[float], list[float]]:
        options = []
        nodes = self.nodes(data=True)
        for i in range(self.max_depth):
            options_for_box = []
            node = nodes[i]
            load = node['load']
            if trans:
                empty_slots = int(round((self.box_size - load) / self.slot_size, 0))
                for j in range(0, empty_slots + 1):
                    if node['trans']:
                        options_for_box.append((j * self.slot_size, True))
                    else:
                        options_for_box.append((j * self.slot_size, True))
                        options_for_box.append((j * self.slot_size, False))
            else:
                options_for_box.append(1)
                if load == 0:
                    options_for_box.append(0)

            options.append(options_for_box)
        for combination in product(*options):
            if trans:
                combination2 = [
                    (self.box_size - self.nodes[i]['load'] - combination[i][0],
                     not combination[i][1] or nodes[i]['trans'])
                    for i in range(self.max_depth)]
                yield list(combination), combination2
            else:
                combination2 = [
                    (max(1 - combination[i], self.nodes[i]['load']), False)
                    for i in range(self.max_depth)]
                yield [(element, False) for element in combination], combination2

    def get_loads(self, trans=True) -> list[tuple[float, bool]] | list[bool]:
        nodes = self.nodes(data=True)
        loads = []
        for node in nodes:
            load = (node[1]['load'], node[1]['trans'] if node[1]['trans'] is not None else False) if trans else node[1]['load']
            loads.append(load)
        return loads

    def merge(self, other: ExtendedDT, box: int, query):
        # print("merging:")
        # self.print_tree()
        # print("with:")
        # other.print_tree()
        # print(f"around query {query} at box: {box}")
        rotated_subtree = other.get_subtree(box)
        rotated_subtree = ExtendedDT(rotated_subtree)
        # print("rotated:")
        # rotated_subtree.print_tree()
        for q in list(other.nodes[box]['qs']):
            if box > 0 and q in other.nodes[box - 1]['qs'] or other.nodes[box]['qs'][q]['c'] <= self.nodes[box]['qs'][query]['c']:
                rotated_subtree.remove_query(q)
            else:
                other.nodes[box]['qs'].pop(q)
        for i in range(box + 1):
            self.nodes[i]['load'] = other.nodes[i]['load']
            self.nodes[i]['qs'] = self.nodes[i]['qs'] | other.nodes[i]['qs']
            self.nodes[i]['trans'] = self.nodes[i]['trans'] if self.nodes[i]['trans'] is not None else other.nodes[i]['trans']
            self.nodes[i]['first'] = self.nodes[i]['first'] if self.nodes[i]['first'] is not None else other.nodes[i][
                'first']
        # print("aligned:")
        # self.print_tree()
        self.nodes[box]['qs'][query]['right_dts'].append(rotated_subtree)
        # print("result:")
        # self.print_tree()

    def query_sequence(self, box):
        return [node for node in self.nodes[box]['qs']]

    def sum_of(self, f):
        box = self.get_root()
        queries_with_right_right_dts = list(self.nodes(data=True)[box]['qs'].items())
        sub_dts = []
        sum_of_f = 0
        for root_query_with_right_dts in queries_with_right_right_dts:
            sub_dts.extend(root_query_with_right_dts[1]["right_dts"])
            sum_of_f += root_query_with_right_dts[1][f]
        self_copy = copy.deepcopy(self)
        self_copy.remove_node(box)
        if box < self.max_depth - 1:
            sub_dts.append(self_copy)
        return sum_of_f + (
            sum(sub_dt.sum_of(f) for sub_dt in sub_dts) if len(sub_dts) == 0 else 0)

    def print_tree(self, node=None, indent=1):
        """
        Rekurencyjny print drzewa ExtendedDT wraz z poddrzewami right_dts.
        """
        if node is None:
            node = self.get_root()

        attrs = self.nodes[node]
        indent_str = "  " * indent
        print(
            f"{indent_str}- Box {node}: load={attrs.get('load', '?')}, trans={attrs.get('trans', '?')}, first={attrs.get('first', '?')}")

        # Wypisz zapytania i poddrzewa right_dts
        for query, qdata in attrs.get('qs', {}).items():
            print(f"{indent_str}  * Query {query}: cost={qdata['c']}")
            for i, sub_dt in enumerate(qdata.get('right_dts', [])):
                print(f"{indent_str}    - Right DT {i}:")
                sub_dt.print_tree(indent=indent + 3)  # zwiększone wcięcie dla poddrzewa

        # Rekurencyjnie wypisz dzieci w grafie (następcy)
        for child in self.successors(node):
            self.print_tree(child, indent + 1)

