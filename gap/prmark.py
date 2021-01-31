import sys
from pathlib import Path
from time import time
from statistics import mean

from pygraphblas import *

def pagerank(A, d, damping, itermax):
    n = A.nrows
    r = Vector.sparse(FP32, n)
    t = Vector.sparse(FP32, n)
    d.assign_scalar(damping, accum=FP32.DIV)
    r[:] = 1.0 / n
    teleport = (1 - damping) / n
    tol = 1e-4
    rdiff = 1.0
    for i in range(itermax):
        # swap t and r
        temp = t ; t = r ; r = temp
        w = t / d
        r[:] = teleport
        A.mxv(w, out=r, accum=FP32.PLUS, semiring=FP32.PLUS_SECOND,
              desc=TransposeA)
        t -= r
        t.apply(FP32.ABS, out=t)
        rdiff = t.reduce_float()
        print('{}: {:.6f}'.format(i, rdiff))
        if rdiff <= tol:
            break
    return r

if __name__ == '__main__':
    argc = len(sys.argv)

    rounds = int(sys.argv[1]) if argc > 1 else 16
    threads = int(sys.argv[2]) if argc > 2 else None

    if threads is not None:
        options_set(nthreads=threads)

    for subdir in ['road', 'kron', 'twitter', 'urand', 'web']:
        fname = '/GAP/GAP-{}/GAP-{}.grb'.format(subdir, subdir)
        if not Path(fname).exists():
            print('Skipping {} No binfile found at {}'.format(subdir, fname))
            continue

        print('loading {} file.'.format(fname))
        M = Matrix.from_binfile(fname.encode('utf8')).pattern()

        M.options_set(format=lib.GxB_BY_COL)
        M.nvals  # finish compute

        d_out = Vector.sparse(FP32, M.nrows)
        M.reduce_vector(FP32.PLUS_MONOID, out=d_out)

        print ("input graph: nodes: {} edges: {}".format(M.nrows, M.nvals))
        print ('Ranking...')

        timings = []
        for i in range(rounds):
            new_d = d_out.dup()
            start = time()
            result = pagerank(M, new_d, 0.85, 100)
            delta = time() - start
            print('Round {} took {}'.format(i, delta))
            timings.append(delta)
            # resultm = Matrix.sparse(result.type, result.size, 1)
            # resultm[:,0] = result
            # resultm.to_mm(open('pr_{}_{}.mtx'.format(subdir, i), 'a'))

        print('PageRank {} average time {} for {} rounds'.format(subdir, mean(timings), rounds))
