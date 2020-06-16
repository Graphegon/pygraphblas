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
        t.apply(FP32.ABS, out=t)
        rdiff = t.reduce_float()
        print('{}: {:.6f}'.format(i, rdiff))
    return r

argc = len(sys.argv)

rounds = int(sys.argv[1]) if argc > 1 else 16
threads = int(sys.argv[2]) if argc > 2 else None

if threads is not None:
    options_set(nthreads=threads)

if __name__ == '__main__':
    for subdir in ['road', 'kron', 'twitter', 'urand', 'web']:
        fname = '/GAP/GAP-{}/GAP-{}.grb'.format(subdir, subdir)
        if not Path(fname).exists():
            print('Skipping {} No binfile found at {}'.format(subdir, fname))
            continue

        print('loading {} file.'.format(fname))
        A = Matrix.from_binfile(fname.encode('utf8'))
        M = A.pattern(UINT64)
        M.options_set(format=lib.GxB_BY_COL)
        M.nvals  # finish compute
        d_out = M.reduce_vector()
        d_in =  M.reduce_vector(desc=TransposeA)
        edges_added = 0
        if d_out.nvals < M.nrows or d_in.nvals < M.nrows:
            printf ("Matrix has {} empty rows and {} empty cols\n".format(M.nrows - d_out.nvals, M.nrows - d_in.nvals))
            for i in range(M.nrows):
                din = d_in[i]
                dout = d_in[i]
                if din is False or dout is False:
                    edges_added += 1
                    M[i:i] = True

        d_out = Vector.sparse(FP32, A.nrows)
        M.reduce_vector(FP32.PLUS_MONOID, out=d_out)
        M = M.pattern()

        print ("input graph: nodes: {} edges: {}".format(M.nrows, M.nvals))
        print ("diag entries added: {}".format(edges_added))

        print ('Ranking...')

        timings = []
        for i in range(rounds):
            start = time()
            result = pagerank(M, d_out, 0.85, 100)
            delta = time() - start
            print('Round {} took {}'.format(i, delta))
            timings.append(delta)
            # uncomment to dump binary result files:
            # resultm = Matrix.sparse(result.type, result.size, 1)
            # resultm[:,0] = result
            # resultm.to_mm(open('pr_{}_{}.mtx'.format(subdir, i), 'a'))

        print('PageRank {} average time {} for {} rounds'.format(subdir, mean(timings), rounds))
