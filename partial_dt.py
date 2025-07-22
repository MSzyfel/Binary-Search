from decision_tree import DecisionTree
import copy


class PartialDT(DecisionTree):
    def __init__(self, base: any = None, box_size=1, max_depth=1):
        super().__init__()
        if isinstance(base, PartialDT):
            self.box_size = base.box_size
            self.max_depth = base.max_depth

            # Deep copy nodes with their attributes
            for node, data in base.nodes(data=True):
                self.add_node(node, **copy.deepcopy(data))

            # Copy edges
            for u, v, edge_data in base.edges(data=True):
                self.add_edge(u, v, **copy.deepcopy(edge_data))
        else:
            self.box_size = box_size
            self.max_depth = max_depth
            for i in range(max_depth):
                self.add_node(i, w=box_size, qs=[], load=0.0)
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
        sub_dts = root_query_with_sub_dts["sub_dts"]
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
        sub_dts = root_query_with_sub_dts["sub_dts"]
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
                new_qs.append({query: {'sub_dts': sub_dts}})
            else:
                new_qs.append({query: {'sub_dts': []}})
                self.put_query(box_index+1, query, weight, sub_dts)
            box = {'load': min(self.box_size, weight), 'qs': new_qs}
            self.nodes(data=True)[box_index][1] = box
