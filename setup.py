from setuptools import setup
import versioneer

with open("README.md") as f:
    long_description = f.read()


setup(
    name="pygraphblas",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="GraphBLAS Python bindings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
