from graphviz import Digraph, Source
from pygraphblas import Matrix, Vector

def draw_vector(V, name='', rankdir='LR', ioff=0, joff=0):
    g = Digraph(name)
    g.attr(rankdir=rankdir, ranksep='1')
    for i, v in V:
        g.node(str(i+ioff), label='%s:%s' % (str(i), str(v)))
    return g

def draw_graph(M, name='', rankdir='LR', show_weight=True, label_vector=None, ioff=0, joff=0):
    g = Digraph(name)
    g.attr(rankdir=rankdir, ranksep='1')
    for i, j, v in M:
        g.node(str(i+ioff), label=str(label_vector[i]) if label_vector else str(i))
        g.node(str(j+joff), label=str(label_vector[j]) if label_vector else str(j))
        g.edge(str(i+ioff), str(j+joff),
               label=str(v) if show_weight else None)
    return g

def draw_layers(M, name='', rankdir='LR'):
    g = Digraph(name)
    g.attr(rankdir=rankdir, ranksep='1')    
    for l, m in enumerate(M):
        with g.subgraph() as s:
            s.attr(rank='same', rankdir='LR')            
            for i in range(m.nrows):
                si = (l * m.nrows) + i
                s.node(str(si), label=str(i))
                if i < m.nrows-1:
                    ni = si +1
                    s.edge(str(si), str(ni), style='invis')
                    
    with g.subgraph() as s:
        s.attr(rank='same', rankdir='LR')            
        for j in range(M[-1].nrows):
            sj = (len(M) * m.nrows) + j
            s.node(str(sj), label=str(j))
            if j < M[-1].nrows-1:
                nj = sj +1
                s.edge(str(sj), str(nj), style='invis')

    for l, m in enumerate(M):
        for i, j, _ in m:
            si = (l * m.nrows) + i
            sj = ((l + 1) * m.nrows) + j
            g.edge(str(si), str(sj))
    return g

def draw(obj, name='', **kws):
    if isinstance(obj, Matrix):
        return draw_graph(obj, name, **kws)
    if isinstance(obj, Vector):
        return draw_vector(obj, name, **kws)

def draw_op(left, op, right, result):
    ioff = 0
    joff = 0
    def draw(obj, name='', **kws):
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
    g.subgraph(draw(left, name='cluster_left'))
    g.node(op)
    g.subgraph(draw(right, name='cluster_right'))
    g.node('=')
    g.subgraph(draw(result, name='cluster_result'))
    return g
