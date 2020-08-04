from setuptools import setup

setup(
    name='pygraphblas',
    version='3.3.1',
    description='GraphBLAS Python bindings.',
    author='Michel Pelletier',
    packages=['pygraphblas'],
    setup_requires=["pytest-runner", "cffi>=1.0.0"],
    cffi_modules=["pygraphblas/build.py:ffibuilder"],
    install_requires=["cffi>=1.0.0","numba>=0.49.0"],
    # tests_require=["pytest","pytest-cov"],
    # entry_points = {
    #     'rdf.plugins.store': [
    #         'graphblas = pygraphblas.rdflib:GraphBLASStore',
    #         ],
    #     }    
)
