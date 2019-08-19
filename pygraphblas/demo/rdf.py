import csv
from collections import defaultdict
from operator import attrgetter

from pygraphblas import Matrix
from pygraphblas.base import NoValue


class Graph:
    """A graph composed of RDF style triples.  Each unique predicate is
    an adjacency matrix from subjects to objects.

    - add(subj, pred, obj)

    - len()

    - query(subj=None, pred=None, obj=None)

    """
    def __init__(self, weight_type=bool):
        self._weight_type = weight_type
        self._max_index = 16
        self._node_count = 0

        self._node_id = {}  # {node -> id}
        self._id_node = {}  # {id -> node}
        self._edge_graph = defaultdict(
            lambda: Matrix.from_type(
                self._weight_type,
                self._max_index,
                self._max_index)
        )

    def _check_and_resize_graphs(self, node_id):
        for edge, graph in self._edge_graph.items():
            nrows = graph.nrows
            newrows = nrows
            if node_id >= nrows:
                newrows = nrows * 2
            if nrows != newrows:
                graph.resize(newrows, newrows)
                self._max_index = newrows

    def _next_node_id(self):
        self._node_count += 1
        return self._node_count

    def _add_node(self, key):
        self._node_id[key] = id = self._next_node_id()
        self._id_node[id] = key
        return id

    def _get_node_id(self, key):
        return self._node_id[key]

    def _get_or_add_node_id(self, key):
        if key not in self._node_id:
            return self._add_node(key)
        return self._get_node_id(key)

    def add(self, subj, pred, obj, weight=True):
        """Add a triple to the graph with an optional weight.

        """
        sid = self._get_or_add_node_id(subj)
        oid = self._get_or_add_node_id(obj)
        self._check_and_resize_graphs(max(sid, oid))
        self._edge_graph[pred][sid, oid] = weight

    def read_csv(self, fname, **kw):
        """Read a csv file of triples into the graph.

        File rows must contain 3 or 4 values, a subj/pred/obj triple
        and an optional weight.

        """
        with open(fname) as fd:
            rd = csv.reader(fd, **kw)
            for row in rd:
                if row:
                    if 3 <= len(row) <= 4:
                        self.add(*row)
                    else:
                        raise TypeError('Row must be 3 or 4 columns')

    def __len__(self):
        """Returns the number of triples in the graph.

        """
        return sum(map(attrgetter('nvals'), self._edge_graph.values()))

    def query(self, subj=None, pred=None, obj=None):
        """Query the graph for matching triples.

        Subject, predicate, and/or object values can be provided, and
        triples that match the given values will be returned.  Passing
        no values will iterate all triples.

        """
        if subj is not None:         # subj,?,?
            if subj not in self._node_id:
                raise KeyError(subj)
            sid = self._node_id[subj]
            if pred is not None:     # subj,pred,?
                if pred not in self._edge_graph:
                    raise KeyError(pred)
                graph = self._edge_graph[pred]
                if obj is not None:  # subj,pred,obj
                    if obj not in self._node_id:
                        raise KeyError(obj)
                    oid = self._node_id[obj]
                    yield subj, pred, obj, graph[sid, oid]
                else:                # subj,pred,None
                    for oid, weight in zip(*graph[sid].to_lists()):
                        yield subj, pred, self._id_node[oid], weight
            else:
                if obj is not None:  # subj,None,obj
                    if obj not in self._node_id:
                        raise KeyError(obj)
                    oid = self._node_id[obj]
                    for pred, graph in self._edge_graph.items():
                        try:
                            weight = graph[sid,oid]
                            yield subj, pred, obj, weight
                        except NoValue:
                            continue
                else:                # subj,None,None
                    for pred, graph in self._edge_graph.items():
                        try:
                            for oid, weight in zip(*graph[sid].to_lists()):
                                yield subj, pred, self._id_node[oid], weight
                        except NoValue:
                            continue
        elif pred is not None:       # None,pred,?
            if pred not in self._edge_graph:
                raise KeyError(pred)
            graph = self._edge_graph[pred]
            if obj is not None:      # None,pred,obj
                if obj not in self._node_id:
                    raise KeyError(obj)
                oid = self._node_id[obj]
                for sid, weight in zip(*graph[:,oid].to_lists()):
                    yield self._id_node[sid], pred, obj, weight
            else:                    # None,pred,None
                for sid, oid, weight in zip(*graph.to_lists()):
                    yield self._id_node[sid], pred, self._id_node[oid], weight
        elif obj is not None:        # None,None,obj
            if obj not in self._node_id:
                raise KeyError(obj)
            oid = self._node_id[obj]
            for pred, graph in self._edge_graph.items():
                try:
                    for sid, weight in zip(*graph[:,oid].to_lists()):
                        yield self._id_node[sid], pred, obj, weight
                except NoValue:
                    continue
        else:                        # None,None,None
            for pred, graph in self._edge_graph.items():
                for sid, oid, weight in zip(*graph.to_lists()):
                    yield self._id_node[sid], pred, self._id_node[oid], weight


