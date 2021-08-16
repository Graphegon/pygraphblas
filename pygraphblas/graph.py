import csv
from collections import defaultdict, OrderedDict, namedtuple
from operator import attrgetter
from typing import NamedTuple

from pygraphblas import Matrix, Vector
from pygraphblas.base import NoValue


class Relation:
    __slots__ = ["graph", "name", "matrix", "record"]

    def __init__(self, graph, name, matrix):
        self.graph = graph
        self.name = name
        self.matrix = matrix

        class Record(NamedTuple):
            subject: object
            object: object
            weight: object

            def __repr__(self):
                return f"{name}({self.subject} -> {self.object}: {self.weight})"

        self.record = Record

    def __repr__(self):
        return f"<{self.matrix.type.__name__} Relation {self.name}>"

    def __matmul__(self, other):
        name = "_".join([self.name, other.name])
        return self.__class__(self.graph, name, self.matrix.mxm(other.matrix))

    def __getattr__(self, name):
        return getattr(self.matrix, name)

    def __iter__(self):
        gi = self.graph._get_id
        for i, j, v in self.matrix:
            yield self.record(gi(i), gi(j), v)


class Graph:
    """A GraphBLAS backed property graph.

    This object is a container for multiple GraphBLAS matrices.  Each
    matrix stores a property relation between graph edges.

    Each unique relation is stored as an adjacency matrix from
    subjects to objects.  To demonstrate, first create a helper
    function `p()` that will iterate results into a list and "pretty
    print" them.

    >>> import pprint
    >>> p = lambda r: pprint.pprint(sorted(list(r)))

    Now construct a graph:

    >>> G = Graph()

    Relation tuples can be added directly into the Graph with the `+=`
    method.

    >>> G += ('bob', 'friend', 'alice')
    >>> G += ('tim', 'friend', 'bob')

    Or an iterator of relation tuples can be provided:

    >>> G += [('bob', 'coworker', 'jane'), ('alice', 'friend', 'jane')]

    The graph can then be called like `G(...)` to query it.  A query
    consists of three optional arguments for `subject`, `relation`
    and `object`.  The default value for all three is None, which acts
    as a wildcard to matches all values.

    >>> p(G())
    [friend(alice -> jane: True),
     friend(bob -> alice: True),
     coworker(bob -> jane: True),
     friend(tim -> bob: True)]

    Only print relations where `bob` is the subject:

    >>> p(G(subject='bob'))
    [friend(bob -> alice: True), coworker(bob -> jane: True)]

    Only print relations where `coworker` is the relation:

    >>> p(G(relation='coworker'))
    [coworker(bob -> jane: True)]

    Only print relations where `jane` is the object:

    >>> p(G(object='jane'))
    [friend(alice -> jane: True), coworker(bob -> jane: True)]

    Relations are `pygraphblas.Matrix` objects, and are accessible as
    attributes of the graph:

    >>> G.friend
    <BOOL Relation friend>
    >>> G.coworker
    <BOOL Relation coworker>

    Relations can be iterated directly:

    >>> p(list(G.friend))
    [friend(alice -> jane: True),
     friend(bob -> alice: True),
     friend(tim -> bob: True)]

    Relations have GraphBLAS Matrix like behavior, they can be
    matrix multiplied:

    >>> G.friend @ G.friend
    <BOOL Relation friend_friend>

    This returns the "friends of friends" relation:

    >>> p(list(G.friend @ G.friend))
    [friend_friend(bob -> jane: True), friend_friend(tim -> alice: True)]

    "This returns the "coworkers of friends" relation:

    >>> p(list(G.friend @ G.coworker))
    [friend_coworker(tim -> jane: True)]

    """

    def __init__(self, start_id=0):
        self._node_count = start_id
        self.nodes = OrderedDict()  # {node -> id}
        self.ids = OrderedDict()  # {id -> node}
        self.relations = {}

    def _next_node(self):
        self._node_count += 1
        return self._node_count

    def _add_node(self, key):
        self.nodes[key] = id = self._next_node()
        self.ids[id] = key
        return id

    def _get_node(self, key):
        return self.nodes[key]

    def _get_id(self, key):
        return self.ids[key]

    def _get_or_add_node(self, key):
        if key not in self.nodes:
            return self._add_node(key)
        return self._get_node(key)

    def _add(self, subject, relation, object, weight=True):
        """Add a triple to the graph with an optional weight."""
        sid = self._get_or_add_node(subject)
        oid = self._get_or_add_node(object)
        if relation not in self.relations:
            self.relations[relation] = Relation(
                self, relation, Matrix.sparse(type(weight))
            )
        self.relations[relation].matrix[sid, oid] = weight

    def __getattr__(self, name):
        if name not in self.relations:
            return AttributeError(name)
        return self.relations[name]

    def __iadd__(self, relation):
        if isinstance(relation, tuple):
            self._add(*relation)
        elif isinstance(relation, Graph):
            raise TypeError("todo")
        else:
            for i in relation:
                self._add(*i)
        return self

    def __len__(self):
        """Returns the number of triples in the graph."""
        return sum(map(attrgetter("nvals"), self.relations.values()))

    def __call__(self, subject=None, relation=None, object=None, ids=False):
        """Query the graph for matching triples.

        Subject, relation, and/or object values can be provided, and
        triples that match the given values will be returned.  Passing
        no values will iterate all triples.

        """
        if subject is not None:  # subject,?,?
            if subject not in self.nodes:
                raise KeyError(subject)
            sid = self.nodes[subject]
            if relation is not None:  # subject,relation,?
                if relation not in self.relations:
                    raise KeyError(relation)
                rel = self.relations[relation]
                if object is not None:  # subject,relation,object
                    if object not in self.nodes:
                        raise KeyError(object)
                    oid = self.nodes[object]
                    yield rel.record(subject, object, rel.matrix[sid, oid])
                else:  # subject,relation,None
                    for oid, weight in rel.matrix[sid]:
                        yield rel.record(subject, self.ids[oid], weight)
            else:
                if object is not None:  # subject,None,object
                    if object not in self.nodes:
                        raise KeyError(object)
                    oid = self.nodes[object]
                    for relation, rel in self.relations.items():
                        try:
                            weight = rel.matrix[sid, oid]
                            yield rel.record(subject, object, weight)
                        except NoValue:
                            continue
                else:  # subject,None,None
                    for relation, rel in self.relations.items():
                        try:
                            for oid, weight in rel.matrix[sid]:
                                yield rel.record(subject, self.ids[oid], weight)
                        except NoValue:
                            continue
        elif relation is not None:  # None,relation,?
            if relation not in self.relations:
                raise KeyError(relation)
            rel = self.relations[relation]
            if object is not None:  # None,relation,object
                if object not in self.nodes:
                    raise KeyError(object)
                oid = self.nodes[object]
                for sid, weight in rel.matrix[:, oid]:
                    yield rel.record(self.ids[sid], object, weight)
            else:  # None,relation,None
                for sid, oid, weight in rel.matrix:
                    yield rel.record(self.ids[sid], self.ids[oid], weight)

        elif object is not None:  # None,None,object
            if object not in self.nodes:
                raise KeyError(object)
            oid = self.nodes[object]
            for relation, rel in self.relations.items():
                try:
                    for sid, weight in rel.matrix[:, oid]:
                        yield rel.record(self.ids[sid], object, weight)
                except NoValue:
                    continue
        else:  # None,None,None
            for relation, rel in self.relations.items():
                for sid, oid, weight in rel.matrix:
                    yield rel.record(self.ids[sid], self.ids[oid], weight)


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
                    raise TypeError("Row must be 3 or 4 columns")
