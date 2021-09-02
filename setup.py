from setuptools import setup
import os

setup(
    name="pygraphblas",
    version="5.1.7.0",
    description="GraphBLAS Python bindings.",
    author="Michel Pelletier",
    packages=["pygraphblas"],
    setup_requires=["pytest-runner"],
    install_requires=[
        "numpy<1.21",
        "numba",
        "suitesparse-graphblas",
        "scipy",
        "contextvars",
        "mmparse",
        "ssgetpy",
        "lazy-property",
    ],
)
