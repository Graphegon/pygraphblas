import sys
from pathlib import Path
from time import time
from statistics import mean

from pygraphblas import *

def load_sources(subdir):
    fname = 'GAP/GAP-{0}/GAP-{0}_sources.mtx'.format(subdir)
    if not Path(fname).exists():
        raise Exception('No sourcefile for {} found at {}'.format(subdir, fname))
    with open(fname) as f:
        m = Matrix.from_mm(f, UINT64)
    return list(zip(*(iter([i[2]-1 for i in m]),) * 4))

def bc(sources, AT, A):
    n = A.nrows
    ns = len(sources)
    paths      = Matrix.dense (FP32, ns, n, 0)
    frontier   = Matrix.sparse(FP32, ns, n)
    S = []

    for i, s in enumerate(sources):
        paths[   i, sources[i]] = 1
        frontier[i, sources[i]] = 1

    frontier.mxm(
        A,
        out=frontier,
        mask=paths,
        semiring=FP32.PLUS_FIRST,
        desc=descriptor.oocr)

    for depth in range(n):
        if frontier.nvals == 0:
            break
        s = Matrix.sparse(BOOL, ns, n)
        frontier.apply(BOOL.ONE, out=s)
        S.append(s)
        paths.assign_matrix(frontier, accum=FP32.PLUS)
        frontier.mxm(A,
            out=frontier,
            mask=paths,
            semiring=FP32.PLUS_FIRST,
            desc=descriptor.oocr)

    bc = Matrix.dense(FP32, ns, n, 1)
    W = Matrix.sparse(FP32, ns, n)

    for i in range(depth - 1, 0, -1):
        bc.emult(paths, FP32.DIV,
                 out=W,
                 mask=S[i],
                 desc=Replace)
        W.mxm(AT, out=W,
              mask=S[i-1],
              semiring=FP32.PLUS_FIRST,
              desc=Replace)
        W.emult(paths, FP32.TIMES,
                out=bc,
                accum=FP32.PLUS)

    centrality = Vector.dense (FP32, n, -ns)
    bc.reduce_vector(accum=FP32.PLUS,
                     out=centrality,
                     desc=TransposeA)
    return centrality

if __name__ == '__main__':
    argc = len(sys.argv)
    threads = int(sys.argv[1]) if argc > 1 else None

    if threads is not None:
        options_set(nthreads=threads)

    for subdir in ['road', 'kron', 'twitter', 'urand', 'web']:
        fname = 'GAP/GAP-{0}/GAP-{0}.grb'.format(subdir)
        if not Path(fname).exists():
            print('Skipping {} No binfile found at {}'.format(subdir, fname))
            continue

        print('loading {} file.'.format(fname))
        M = Matrix.from_binfile(fname.encode('utf8'))
        MT = M.T

        print('Betweening...')
        timings = []
        sources = load_sources(subdir)
        for i, s in enumerate(sources):
            start = time()
            result = bc(s, MT, M)
            delta = time() - start
            print('Round {} took {}'.format(i, delta))
            timings.append(delta)
            resultm = Matrix.sparse(result.type, result.size, 1)
            resultm[:,0] = result
            resultm.to_mm(open('bc_{}_{}.mtx'.format(subdir, i), 'a'))

        print('BetweenessCentraility {} average time {}'.format(subdir, mean(timings)))
