
from .base import lib

class MatrixFuncs:
    __slots__ = (
        'C',
        'setElement',
        'extractElement',
        'extractTuples',
        'add_op',
        'mult_op',
        'monoid',
        'semiring',
        'assignScalar',
        'invert',
        'neg',
        'abs_',
        'not_',
        )

def build_matrix_type_funcs(typ):
    f = MatrixFuncs()
    if typ == lib.GrB_BOOL:
        c, s, a, m  = ('_Bool', 'BOOL', 'LOR', 'LAND')
    elif typ == lib.GrB_INT8:
        c, s, a, m  = ('int8_t', 'INT8', 'PLUS', 'TIMES')
    elif typ == lib.GrB_INT16:
        c, s, a, m  = ('int16_t', 'INT16', 'PLUS', 'TIMES')
    elif typ == lib.GrB_INT32:
        c, s, a, m  = ('int32_t', 'INT32', 'PLUS', 'TIMES')
    elif typ == lib.GrB_INT64:
        c, s, a, m  = ('int64_t', 'INT64', 'PLUS', 'TIMES')
    elif typ == lib.GrB_FP64:
        c, s, a, m  = ('double', 'FP64', 'PLUS', 'TIMES')
    elif typ == lib.GrB_FP32:
        c, s, a, m  = ('float', 'FP32', 'PLUS', 'TIMES')
    f.C = c
    f.setElement = getattr(lib, 'GrB_Matrix_setElement_{}'.format(s))
    f.extractElement = getattr(lib, 'GrB_Matrix_extractElement_{}'.format(s))
    f.extractTuples = getattr(lib, 'GrB_Matrix_extractTuples_{}'.format(s))
    f.add_op = getattr(lib, 'GrB_PLUS_{}'.format(s))
    f.mult_op = getattr(lib, 'GrB_TIMES_{}'.format(s))
    f.monoid = getattr(lib, 'GxB_{}_{}_MONOID'.format(a, s))
    f.semiring = getattr(lib, 'GxB_{}_{}_{}'.format(a, m, s))
    f.assignScalar = getattr(lib, 'GrB_Matrix_assign_{}'.format(s))
    f.invert = getattr(lib, 'GrB_MINV_{}'.format(s))
    f.neg = getattr(lib, 'GrB_AINV_{}'.format(s))
    f.abs_ = getattr(lib, 'GxB_ABS_{}'.format(s))
    f.not_ = getattr(lib, 'GxB_LNOT_{}'.format(s))
    return f


class VectorFuncs:
    __slots__ = (
        'C',
        'setElement',
        'extractElement',
        'extractTuples',
        'add_op',
        'mult_op',
        'semiring',
        'assignScalar',
        'invert',
        'neg',
        'abs_',
        'not_',
        )

def build_vector_type_funcs(typ):
    f = VectorFuncs()
    if typ == lib.GrB_BOOL:
        c, s, a, m  = ('_Bool', 'BOOL', 'LOR', 'LAND')
    elif typ == lib.GrB_INT8:
        c, s, a, m  = ('int8_t', 'INT8', 'PLUS', 'TIMES')
    elif typ == lib.GrB_INT16:
        c, s, a, m  = ('int16_t', 'INT16', 'PLUS', 'TIMES')
    elif typ == lib.GrB_INT32:
        c, s, a, m  = ('int32_t', 'INT32', 'PLUS', 'TIMES')
    elif typ == lib.GrB_INT64:
        c, s, a, m  = ('int64_t', 'INT64', 'PLUS', 'TIMES')
    elif typ == lib.GrB_FP64:
        c, s, a, m  = ('double', 'FP64', 'PLUS', 'TIMES')
    elif typ == lib.GrB_FP32:
        c, s, a, m  = ('float', 'FP32', 'PLUS', 'TIMES')
    f.C = c
    f.setElement = getattr(lib, 'GrB_Vector_setElement_{}'.format(s))
    f.extractElement = getattr(lib, 'GrB_Vector_extractElement_{}'.format(s))
    f.extractTuples =  getattr(lib, 'GrB_Vector_extractTuples_{}'.format(s))
    f.add_op = getattr(lib, 'GrB_PLUS_{}'.format(s))
    f.mult_op = getattr(lib, 'GrB_TIMES_{}'.format(s))
    f.semiring = getattr(lib, 'GxB_{}_{}_{}'.format(a, m, s))
    f.assignScalar = getattr(lib, 'GrB_Vector_assign_{}'.format(s))
    f.invert = getattr(lib, 'GrB_AINV_{}'.format(s))
    f.abs_ = getattr(lib, 'GxB_ABS_{}'.format(s))
    return f

class ScalarFuncs:
    __slots__ = (
        'C',
        'setElement',
        'extractElement',
        )
def build_scalar_type_funcs(typ):
    f = ScalarFuncs()
    if typ == lib.GrB_BOOL:
        c, s  = ('_Bool', 'BOOL')
    elif typ == lib.GrB_INT8:
        c, s  = ('int8_t', 'INT64')
    elif typ == lib.GrB_INT16:
        c, s  = ('int16_t', 'INT64')
    elif typ == lib.GrB_INT32:
        c, s  = ('int32_t', 'INT64')
    elif typ == lib.GrB_INT64:
        c, s  = ('int64_t', 'INT64')
    elif typ == lib.GrB_FP64:
        c, s  = ('double', 'FP64')
    elif typ == lib.GrB_FP32:
        c, s  = ('float', 'FP32')
    f.C = c
    f.setElement = getattr(lib, 'GxB_Scalar_setElement_{}'.format(s))
    f.extractElement = getattr(lib, 'GxB_Scalar_extractElement_{}'.format(s))
    return f
