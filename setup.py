from setuptools import setup
import os

setup(
    name="pygraphblas",
    version="5.1.5.1",
    description="GraphBLAS Python bindings.",
    author="Michel Pelletier",
    packages=["pygraphblas"],
    setup_requires=["pytest-runner"],
    install_requires=[
        "numba",
        "suitesparse-graphblas",
        "scipy",
        "contextvars",
        "mmparse",
        "ssgetpy",
        "lazy-property",
    ],
)
