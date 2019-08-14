import csv
from collections import defaultdict
from operator import attrgetter

from pygraphblas import Matrix


class Graph:
    def __init__(self, weight_type=bool):
        self.weight_type = weight_type
        self.max_index = 16
        self.node_count = 0

        self.node_id = {}  # {node -> id}
        self.id_node = {}  # {id -> node}
        self.edge_graph = defaultdict(
            lambda: Matrix.from_type(
                self.weight_type,
                self.max_index,
                self.max_index)
        )

    def check_and_resize_graphs(self, node_id):
        for edge, graph in self.edge_graph.items():
            nrows = graph.nrows
            newrows = nrows
            if node_id >= nrows:
                newrows = nrows * 2
            if nrows != newrows:
                graph.resize(newrows, newrows)
                self.max_index = newrows

    def next_node_id(self):
        self.node_count += 1
        return self.node_count

    def add_node(self, key):
        self.node_id[key] = id = self.next_node_id()
        self.id_node[id] = key
        return id

    def get_node_id(self, key):
        return self.node_id[key]

    def get_or_add_node_id(self, key):
        if key not in self.node_id:
            return self.add_node(key)
        return self.get_node_id(key)

    def add(self, subj, pred, obj, weight=True):
        sid = self.get_or_add_node_id(subj)
        oid = self.get_or_add_node_id(obj)
        self.check_and_resize_graphs(max(sid, oid))
        self.edge_graph[pred][sid, oid] = weight

    def read_csv(self, fname, **kw):
        with open(fname) as fd:
            rd = csv.reader(fd, **kw)
            for row in rd:
                if row:
                    if 3 <= len(row) <= 4:
                        self.add(*row)
                    else:
                        raise TypeError('Row must be 3 or 4 columns')

    def __len__(self):
        return sum(map(attrgetter('nvals'), g.edge_graph.values()))

    def query(self, subj=None, pred=None, obj=None):
        if subj is not None:         # subj,?,?
            if subj not in self.node_id:
                raise KeyError(subj)
            if pred is not None:     # subj,pred,?
                if pred not in self.edge_graph:
                    raise KeyError(pred)
                if obj is not None:  # subj,pred,obj
                    if obj not in self.node_id:
                        raise KeyError(obj)
                    sid = self.node_id[subj]
                    oid = self.node_id[obj]
                else:                # subj,pred,None
                    pass
            else:
                if obj is not None:  # subj,None,obj
                    if obj not in self.node_id:
                        raise KeyError(obj)
                else:                # subj,None,None
                    pass
        elif pred is not None:       # None,pred,?
            if pred not in self.edge_graph:
                raise KeyError(pred)
            if obj is not None:      # None,pred,obj
                if obj not in self.node_id:
                    raise KeyError(obj)
            else:                    # None,pred,None
                pass
        elif obj is not None:        # None,None,obj
            if obj not in self.node_id:
                raise KeyError(obj)
        else:                        # None,None,None
            pass

