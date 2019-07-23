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

RUN curl -s -L  http://faculty.cse.tamu.edu/davis/GraphBLAS/GraphBLAS-3.0.1-beta1.tar.gz | tar -xz && \
     cd GraphBLAS-3.0.1-beta1 && \
#    sed -i 's/^\/\/ #undef NDEBUG/#undef NDEBUG/g' Source/GB.h && \
#    sed -i 's/^\/\/ #define GB_PRINT_MALLOC 1/#define GB_PRINT_MALLOC 1/g' Source/GB.h && \
    make library \
#    CMAKE_OPTIONS='-DCMAKE_BUILD_TYPE=Debug' \
    && make install

RUN git clone --branch 22July2019 https://github.com/GraphBLAS/LAGraph.git && \
    cd LAGraph && \
    make library \
#    CMAKE_OPTIONS='-DCMAKE_BUILD_TYPE=Debug' \
    && make install

RUN ldconfig

RUN conda install -yq datashader pytest ipdb

ADD . /pygraphblas
WORKDIR /pygraphblas
# RUN python setup.py clean
RUN python setup.py install
RUN chown -R jovyan:users .

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID