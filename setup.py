from setuptools import setup

setup(
    name='pygraphblas',
    version='0.0.1',
    description='GraphBLAS Python bindings.',
    author='Michel Pelletier',
    packages=['pygraphblas'],
    setup_requires=["pytest-runner", "cffi>=1.0.0"],
    cffi_modules=["pygraphblas/build.py:ffibuilder"],
    install_requires=["cffi>=1.0.0"],
    tests_require=["pytest"],
    # entry_points = {
    #     'rdf.plugins.store': [
    #         'graphblas = pygraphblas.rdflib:GraphBLASStore',
    #         ],
    #     }    
)
