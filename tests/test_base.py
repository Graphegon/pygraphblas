import pytest

from pygraphblas import *

def test_options_set():
    opts = options_get()

    iz = lambda name, typ: isinstance(opts.get(name), typ)

    assert iz("nthreads", int)
    assert iz("chunk", float)
    assert iz("burble", int)
    assert iz("format", int)
    assert iz("hyper_switch", float)
    assert iz("bitmap_switch", list)

    assert opts["burble"] == 0

    options_set(nthreads=4)
    options_set(chunk=4096)
    options_set(burble=1)
    options_set(format=lib.GxB_BY_COL)
    options_set(hyper_switch=1.0)
    options_set(bitmap_switch=[1, 2, 3, 4, 5, 6, 7, 8])

    news = options_get()
    ez = lambda name, v: news.get(name) == v

    assert ez("nthreads", 4)
    assert ez("chunk", 4096)
    assert ez("burble", 1)
    assert ez("format", lib.GxB_BY_COL)
    assert ez("hyper_switch", 1.0)
    assert ez("bitmap_switch", [1, 2, 3, 4, 5, 6, 7, 8])

    options_set(**opts)
    assert opts == options_get()

from pygraphblas import *

def test_get_version():
    v = get_version()
    assert isinstance(v, tuple)
    assert isinstance(v[0], int)
    assert isinstance(v[1], int)
    
