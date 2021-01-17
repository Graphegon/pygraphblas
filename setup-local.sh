#!/bin/bash

# this script is to setup a locally installed copy of SuiteSparse and
# pygraphblas into a local venv style virtual environment.

# you will need
# apt-get update && apt-get install -yq cmake make wget libpython3-dev python3-pip libreadline-dev llvm-10-dev git python3-virtualenv

SS_RELEASE=v4.0.0draft5
SS_BURBLE=0
SS_COMPACT=1

mkdir -p local/build
cd local/build
git clone https://github.com/DrTimothyAldenDavis/GraphBLAS.git --depth 1 --branch ${SS_RELEASE}

cd GraphBLAS
cmake . -DGB_BURBLE=${SS_BURBLE} -DGBCOMPACT=${SS_COMPACT} && make -j8 && make DESTDIR=.. install

cd ..

python3 -m virtualenv --python=python3 venv

git clone https://github.com/Graphegon/pygraphblas.git --depth 1 --branch Graphegon/${SS_RELEASE}
cd pygraphblas

cat  <<'EOF' >> libpatch.diff
diff --git a/pygraphblas/build.py b/pygraphblas/build.py
index 72b42e3..f448892 100644
--- a/pygraphblas/build.py
+++ b/pygraphblas/build.py
@@ -40,6 +40,8 @@ def build_ffi():
         "_pygraphblas",
         source,
         libraries=["graphblas"],
+        include_dirs=['../usr/local/include'],
+        library_dirs=['../usr/local/lib'],
         extra_compile_args=[
             "-std=c11",
             "-lm",
EOF

git apply libpatch.diff

virtualenv --python=python3 venv
. venv/bin/activate

pip3 install -r minimal-requirements.txt
python3 setup.py install
