from setuptools import setup
import os

setup(
    name='pygraphblas',
    version='4.0.0',
    description='GraphBLAS Python bindings.',
    author='Michel Pelletier',
    packages=['pygraphblas'],
    setup_requires=["pytest-runner", "cffi>=1.0.0"],
    cffi_modules=["pygraphblas/build.py:ffibuilder"],
    install_requires=["cffi>=1.0.0", "numpy>=1.15", "numba", "scipy"],
    # tests_require=["pytest","pytest-cov"],
    # entry_points = {
    #     'rdf.plugins.store': [
    #         'graphblas = pygraphblas.rdflib:GraphBLASStore',
    #         ],
    #     }    
)
