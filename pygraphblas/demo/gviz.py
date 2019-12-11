from graphviz import Digraph, Source

def draw_graph(M, name='', rankdir='LR', show_weight=True, label_vector=None):
    g = Digraph(name)
    g.attr(rankdir=rankdir, ranksep='1')
    for i, j, v in M:
        g.node(str(i), label=str(label_vector[i]) if label_vector else str(i))
        g.node(str(j), label=str(label_vector[j]) if label_vector else str(j))
        g.edge(str(i), str(j),
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
                if i != m.nrows-1:
                    ni = si +1
                    s.edge(str(si), str(ni), style='invis')
        if l != len(M)-1:
            for i, j, _ in m:
                si = (l * m.nrows) + i
                sj = ((l + 1) * m.nrows) + j
                g.edge(str(si), str(sj))
    return g
