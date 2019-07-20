ARG BASE_CONTAINER=jupyter/base-notebook
FROM $BASE_CONTAINER

USER root

# Install all OS dependencies for fully functional notebook server
RUN apt-get update && apt-get install -yq --no-install-recommends \
    build-essential \
    emacs \
    git \
    inkscape \
    jed \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    netcat \
    pandoc \
    python-dev \
    texlive-fonts-extra \
    texlive-fonts-recommended \
    texlive-generic-recommended \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-xetex \
    tzdata \
    unzip \
    nano \
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
# RUN curl -s -L http://faculty.cse.tamu.edu/davis/GraphBLAS/GraphBLAS-2.3.3.tar.gz | \
RUN curl -s -L https://uc8d09ad789b3623852c0c1398be.dl.dropboxusercontent.com/cd/0/get/AlCx86NG4AnB0FBf8NdINTpZbAyh9L1Y_rE2naz3hY-OMYpxd4t7CM_llpwDZmQxAmIYXG-r-34NcNDsgBKgkRAYGC8QyhQnMk7K1tC3V7u9bw/file# | \
#    tar zxvf - && cd GraphBLAS && \
    tar zxvf - && cd GraphBLAS_Jun11_2019 && \
#    sed -i 's/^\/\/ #undef NDEBUG/#undef NDEBUG/g' Source/GB.h && \
#    sed -i 's/^\/\/ #define GB_PRINT_MALLOC 1/#define GB_PRINT_MALLOC 1/g' Source/GB.h && \
    make library \
#    CMAKE_OPTIONS='-DCMAKE_BUILD_TYPE=Debug' \
    && make install

RUN git clone https://github.com/GraphBLAS/LAGraph.git && \
    cd LAGraph && \
    git checkout 7a21aa5 && \
    make library \
#    CMAKE_OPTIONS='-DCMAKE_BUILD_TYPE=Debug' \
    && make install

RUN ldconfig

RUN conda install -yq datashader pytest

ADD . /pygraphblas
WORKDIR /pygraphblas
RUN python setup.py clean
RUN python setup.py install
RUN chown -R jovyan:users .

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID