from graphviz import Digraph, Source
from pygraphblas import Matrix, Vector


def _str(s, label_width):
    return str(s)[:label_width]


def draw_vector(V, name="", rankdir="LR", ioff=0, joff=0):
    g = Digraph(name)
    g.attr(rankdir=rankdir, ranksep="1")
    for i, v in V:
        g.node(str(i + ioff), label="%s:%s" % (str(i), str(v)))
    return g


def draw_graph(
    M,
    name="",
    rankdir="LR",
    show_weight=True,
    concentrate=True,
    label_vector=None,
    label_width=None,
    size_vector=None,
    size_scale=1.0,
    ioff=0,
    joff=0,
    filename=None,
    size=None,
):
    g = Digraph(name)
    g.attr(rankdir=rankdir, ranksep="1", overlap="false", concentrate="true")
    if size is not None:
        g.attr(size=size)
    for i, j, v in M:
        size = _str(size_vector[i] * size_scale, label_width) if size_vector else "0.5"
        ilabel = _str(label_vector[i], label_width) if label_vector else str(i)
        jlabel = _str(label_vector[j], label_width) if label_vector else str(j)
        vlabel = _str(v, label_width) if show_weight else None

        g.node(str(i + ioff), width=size, height=size, label=ilabel)
        g.node(str(j + joff), width=size, height=size, label=jlabel)
        w = str(int(v))
        g.edge(str(i + ioff), str(j + joff),
               label=vlabel, weight=w, tooltip=vlabel, len=str(0.3))
    if filename is not None:
        g.render(filename, format='png')
    return g


def draw_layers(M, name="", rankdir="LR", label_width=None):
    g = Digraph(name)
    g.attr(rankdir=rankdir, ranksep="1")
    for l, m in enumerate(M):
        with g.subgraph() as s:
            s.attr(rank="same", rankdir="LR")
            for i in range(m.nrows):
                si = (l * m.nrows) + i
                s.node(str(si), label=_str(i, label_width), width="0.5")
                if i < m.nrows - 1:
                    ni = si + 1
                    s.edge(str(si), str(ni), style="invis")

    with g.subgraph() as s:
        s.attr(rank="same", rankdir="LR")
        for j in range(M[-1].nrows):
            sj = (len(M) * m.nrows) + j
            s.node(str(sj), label=_str(j, label_width), width="0.5")
            if j < M[-1].nrows - 1:
                nj = sj + 1
                s.edge(str(sj), str(nj), style="invis")

    for l, m in enumerate(M):
        for i, j, _ in m:
            si = (l * m.nrows) + i
            sj = ((l + 1) * m.nrows) + j
            g.edge(str(si), str(sj))
    return g


def draw(obj, name="", **kws):
    if isinstance(obj, Matrix):
        return draw_graph(obj, name, **kws)
    if isinstance(obj, Vector):
        return draw_vector(obj, name, **kws)


def draw_op(left, op, right, result):
    ioff = 0
    joff = 0

    def draw(obj, name="", **kws):
        nonlocal ioff, joff
        if isinstance(obj, Matrix):
            ioff += obj.nrows
            joff += obj.ncols
            return draw_graph(obj, name=name, ioff=ioff, joff=joff)
        if isinstance(obj, Vector):
            ioff += obj.size
            joff += obj.size
            return draw_vector(obj, name=name, ioff=ioff, joff=joff)

    g = Digraph()
    g.subgraph(draw(left, name="cluster_left"))
    g.node(op, width="0.5")
    g.subgraph(draw(right, name="cluster_right"))
    g.node("=", width="0.5")
    g.subgraph(draw(result, name="cluster_result"))
    return g
