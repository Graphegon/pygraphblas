#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat manylinux2010_x86_64 -w /io/wheelhouse/
    fi
}

yum install -y cmake make gcc git openmp-dev
git clone --depth=1 --branch=v4.0.3 https://github.com/DrTimothyAldenDavis/GraphBLAS.git
cd GraphBLAS/build
cmake .. && make -j8 && make install
ldconfig

# Compile wheels
for PYBIN in /opt/python/cp3[678]*/bin; do
    "${PYBIN}/pip" install -r /io/manylinux-requirements.txt
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYBIN in /opt/python/cp3[678]*/bin/; do
    "${PYBIN}/pip" install pygraphblas --no-index -f /io/wheelhouse
    (cd "$HOME"; "${PYBIN}/pytest" pygraphblas)
done
