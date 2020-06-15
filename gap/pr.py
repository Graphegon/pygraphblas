from pygraphblas import *

M = Matrix.random(FP32, 7, 7, 30, no_diagonal=True, make_pattern=True, seed=42)

def pagerank3f(A, damping, itermax):
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
    return r
    

pr = pagerank3f(M, 0.35, 3)
