ARG BASE_CONTAINER=python:3.7
FROM $BASE_CONTAINER

USER root

# Install all OS dependencies for fully functional notebook server
RUN apt-get update && apt-get install -yq --no-install-recommends \
    build-essential \
    git \
    netcat \
    python-dev \
    tzdata \
    unzip \
    make \
    cmake \
    curl \
    sudo \
    libreadline-dev \
    tmux \
    zile \
    zip \
    vim \
    gawk \
    wget \
    m4 \
    && rm -rf /var/lib/apt/lists/*

# get GraphBLAS, compile with debug symbols

RUN curl -s -L  http://faculty.cse.tamu.edu/davis/GraphBLAS/GraphBLAS-3.1.1.tar.gz | tar -xz && \
     cd GraphBLAS-3.1.1 && \
#    sed -i 's/^\/\/ #undef NDEBUG/#undef NDEBUG/g' Source/GB.h && \
#    sed -i 's/^\/\/ #define GB_PRINT_MALLOC 1/#define GB_PRINT_MALLOC 1/g' Source/GB.h && \
    make library \
#    CMAKE_OPTIONS='-DCMAKE_BUILD_TYPE=Debug' \
    && make install
RUN cd .. && /bin/rm -Rf GraphBLAS

RUN git clone --branch 22July2019 https://github.com/GraphBLAS/LAGraph.git && \
    cd LAGraph && \
    make library \
#    CMAKE_OPTIONS='-DCMAKE_BUILD_TYPE=Debug' \
    && make install
RUN cd .. && /bin/rm -Rf LAGraph

RUN ldconfig

ADD . /pygraphblas
WORKDIR /pygraphblas
RUN python setup.py clean
RUN python setup.py develop
RUN pip install pytest pytest-cov ipdb
