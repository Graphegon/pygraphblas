ARG BASE_CONTAINER=ubuntu:20.04
FROM ${BASE_CONTAINER}

USER root

RUN apt-get update && apt-get install -yq --no-install-recommends \
    build-essential \
    libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

ADD . /
WORKDIR /
    
RUN python setup.py clean
RUN python setup.py install
RUN pip install -r minimal-requirements.txt
RUN ldconfig
