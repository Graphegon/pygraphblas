import sys
from pathlib import Path
from time import time
from statistics import mean

from pygraphblas import *

def pagerank(A, damping, itermax):
    n = A.nrows
    r = Vector.sparse(FP32, n)
    t = Vector.sparse(FP32, n)
    d = A.reduce_vector()
    with Accum(FP32.DIV):
        d[:] = damping
    r[:] = 1.0 / n
    teleport = (1 - damping) / n
    tol = 1e-4
    rdiff = 1.0
    for i in range(itermax):
        if rdiff <= tol:
            break
        temp = t ; t = r ; r = temp
        w = t / d
        r[:] = teleport
        A.mxv(w,
              out=r,
              accum=FP32.PLUS,
              semiring=FP32.PLUS_SECOND,
              desc=TransposeA)
        t -= r
        t = abs(t)
        rdiff = t.reduce_float()
        print('rdiff {}'.format(rdiff))
    print('PR took {} iterations'.format(i))
    return r

argc = len(sys.argv)

rounds = int(sys.argv[1]) if argc > 1 else 16
threads = int(sys.argv[2]) if argc > 2 else None

if threads is not None:
    options_set(nthreads=threads)

if __name__ == '__main__':
    for subdir in ['road', 'kron', 'twitter', 'urand', 'web']:
        fname = 'GAP/GAP-{}/GAP-{}.grb'.format(subdir, subdir)
        if not Path(fname).exists():
            print('Skipping {} No binfile found at {}'.format(subdir, fname))
            continue
        
        print('loading {} file.'.format(fname))
        M = Matrix.from_binfile(fname.encode('utf8'))

        print('Ranking...')

        timings = []
        for i in range(rounds):
            start = time()
            pr = pagerank(M, 0.85, 100)
            delta = time() - start
            print('Round {} took {}'.format(i, delta))
            timings.append(delta)
            pr.to_mm(open('pr_{}_{}.mtx'.format(subdir, i), 'a'))

        print('PageRank {} average time {} for {} rounds'.format(subdir, mean(timings), rounds))
