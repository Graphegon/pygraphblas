
from .base import lib

def build_matrix_type_funcs():
    _type_funcs = {}
    for c, t, s, a, m in (('_Bool', lib.GrB_BOOL, 'BOOL', 'LOR', 'LAND'),
                          ('int64_t', lib.GrB_INT64, 'INT64', 'PLUS', 'TIMES'),
                          ('double', lib.GrB_FP64, 'FP64', 'PLUS', 'TIMES'),
                          ('float', lib.GrB_FP32, 'FP32', 'PLUS', 'TIMES')):
        _type_funcs[t] = {
            'C': c,
            'setElement': getattr(lib, 'GrB_Matrix_setElement_{}'.format(s)),
            'extractElement': getattr(lib, 'GrB_Matrix_extractElement_{}'.format(s)),
            'extractTuples': getattr(lib, 'GrB_Matrix_extractTuples_{}'.format(s)),
            'add_op': getattr(lib, 'GrB_PLUS_{}'.format(s)),
            'mult_op': getattr(lib, 'GrB_TIMES_{}'.format(s)),
            'monoid': getattr(lib, 'GxB_{}_{}_MONOID'.format(a, s)),
            'semiring': getattr(lib, 'GxB_{}_{}_{}'.format(a, m, s)),
            'assignScalar': getattr(lib, 'GrB_Matrix_assign_{}'.format(s)),
            'invert': getattr(lib, 'GrB_MINV_{}'.format(s)),
            'neg': getattr(lib, 'GrB_AINV_{}'.format(s)),
            'abs': getattr(lib, 'GxB_ABS_{}'.format(s)),
            'not': getattr(lib, 'GxB_LNOT_{}'.format(s)),
        }
    return _type_funcs


def build_vector_type_funcs():
    _type_funcs = {}
    for c, t, s, a, m in (('_Bool', lib.GrB_BOOL, 'BOOL', 'LOR', 'LAND'),
                          ('int64_t', lib.GrB_INT64, 'INT64', 'PLUS', 'TIMES'),
                          ('double', lib.GrB_FP64, 'FP64', 'PLUS', 'TIMES'),
                          ('float', lib.GrB_FP32, 'FP32', 'PLUS', 'TIMES')):
        _type_funcs[t] = {
            'C': c,
            'setElement': getattr(lib, 'GrB_Vector_setElement_{}'.format(s)),
            'extractElement': getattr(lib, 'GrB_Vector_extractElement_{}'.format(s)),
            'extractTuples': getattr(lib, 'GrB_Vector_extractTuples_{}'.format(s)),
            'add_op': getattr(lib, 'GrB_PLUS_{}'.format(s)),
            'mult_op': getattr(lib, 'GrB_TIMES_{}'.format(s)),
            'semiring': getattr(lib, 'GxB_{}_{}_{}'.format(a, m, s)),
            'assignScalar': getattr(lib, 'GrB_Vector_assign_{}'.format(s)),
            'invert': getattr(lib, 'GrB_AINV_{}'.format(s)),
            'abs': getattr(lib, 'GxB_ABS_{}'.format(s)),
        }
    return _type_funcs
