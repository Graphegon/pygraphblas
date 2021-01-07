import pytest

from pygraphblas import *

pytest.mark.skip()


def _test_udt():
    class BF(Type):

        members = ["double w", "uint64_t h", "uint64_t pi"]
        one = (lib.INFINITY, lib.UINT64_MAX, lib.UINT64_MAX)

        @binop(boolean=True)
        def EQ(z, x, y):
            if not x.w == y.w and x.h == y.h and x.pi == y.pi:
                z = True
            else:
                z = False

        @binop()
        def PLUS(z, x, y):
            if (
                x.w < y.w
                or x.w == y.w
                and x.h < y.h
                or x.w == y.w
                and x.h == y.h
                and x.pi < y.pi
            ):
                if z.w != x.w and z.h != x.h and z.pi != x.pi:
                    z.w = x.w
                    z.h = x.h
                    z.pi = x.pi
            else:
                z.w = y.w
                z.h = y.h
                z.pi = y.pi

        @binop()
        def TIMES(z, x, y):
            z.w = x.w + y.w
            z.h = x.h + y.h
            if x.pi != lib.UINT64_MAX and y.pi != 0:
                z.pi = y.pi
            else:
                z.pi = x.pi

    BF_monoid = BF.new_monoid(BF.PLUS, BF.one)
    BF_semiring = BF.new_semiring(BF_monoid, BF.TIMES)

    def shortest_path(matrix, start):
        n = matrix.nrows
        v = Vector.sparse(matrix.type, n)
        for i, j, k in matrix:
            if i == j:
                matrix[i, j] = (0, 0, 0)
            else:
                matrix[i, j] = (k[0], 1, i)
        v[start] = (0, 0, 0)
        with BF_semiring, Accum(BF.PLUS):
            for _ in range(matrix.nrows):
                w = v.dup()
                v @= matrix
                if w.iseq(v):
                    break
        return v

    A = Matrix.sparse(BF, 6, 6)
    A[0, 1] = (9.0, 0, 0)
    A[0, 3] = (3.0, 0, 0)
    A[1, 2] = (8.0, 0, 0)
    A[3, 4] = (6.0, 0, 0)
    A[3, 5] = (1.0, 0, 0)
    A[4, 2] = (4.0, 0, 0)
    A[1, 5] = (7.0, 0, 0)
    A[5, 4] = (2.0, 0, 0)

    v = shortest_path(A, 0)

    assert v.to_lists() == [
        [0, 1, 2, 3, 4, 5],
        [(0.0, 0, 0), (9.0, 1, 0), (10.0, 4, 4), (3.0, 1, 0), (6.0, 3, 5), (4.0, 2, 3)],
    ]


def test_log_semiring():
    from math import log, log1p, exp

    class Log32(FP32):
        @binary_op(FP32)
        def PLUS(x, y):
            return x + log1p(exp(y - x))

        @binary_op(FP32)
        def TIMES(x, y):
            return x + y

        @classmethod
        def from_value(cls, value):
            return log(value)

        @classmethod
        def to_value(cls, data):
            return exp(data)

    A = Matrix.sparse(Log32, 6, 6)
    A[0, 1] = 1 / 9.0
    A[0, 3] = 1 / 3.0
    A[1, 2] = 1 / 8.0
    A[3, 4] = 1 / 6.0
    A[3, 5] = 1 / 1.0
    A[4, 2] = 1 / 4.0
    A[1, 5] = 1 / 7.0
    A[5, 4] = 1 / 2.0

    Log32_monoid = Log32.new_monoid(Log32.PLUS, Log32.one)
    Log32_semiring = Log32.new_semiring(Log32_monoid, Log32.TIMES)

    with Log32_semiring:
        B = A @ A

    assert B.to_lists() == [
        [0, 0, 0, 1, 3, 3, 5],
        [2, 4, 5, 4, 2, 4, 2],
        [
            0.01388888825858143,
            0.055555553245953966,
            0.34920633498203557,
            0.0714285835851032,
            0.041666665602164574,
            0.49999999904767284,
            0.12499999928575464,
        ],
    ]
