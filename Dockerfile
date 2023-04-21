# Docker file for the sensitivity analysis program.
# This file first includes Gurobi, and then adds 
# Stormpy with the Python program next to it.

FROM ubuntu:20.04 as buildoptimizer
ARG GRB_VERSION=10.0.0
ARG GRB_SHORT_VERSION=10.0

MAINTAINER Thom Badings <thom.badings@ru.nl>

# install gurobi package and copy the files
WORKDIR /opt

RUN apt-get update \
    && apt-get install --no-install-recommends -y\
       ca-certificates  \
       wget \
    && update-ca-certificates \
    && wget -v https://packages.gurobi.com/${GRB_SHORT_VERSION}/gurobi${GRB_VERSION}_linux64.tar.gz \
    && tar -xvf gurobi${GRB_VERSION}_linux64.tar.gz  \
    && rm -f gurobi${GRB_VERSION}_linux64.tar.gz \
    && mv -f gurobi* gurobi \
    && rm -rf gurobi/linux64/docs

# After the file renaming, a clean image is build
FROM python:3.10-slim-bullseye AS packageoptimizer

ARG GRB_VERSION=10.0.0

LABEL vendor="Gurobi"
LABEL version=${GRB_VERSION}

# update system and certificates
RUN apt-get update \
    && apt-get install --no-install-recommends -y\
       ca-certificates  \
       p7zip-full \
       zip \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/gurobi
COPY --from=buildoptimizer /opt/gurobi .

ENV GUROBI_HOME /opt/gurobi/linux64
ENV PATH $PATH:$GUROBI_HOME/bin
ENV LD_LIBRARY_PATH $GUROBI_HOME/lib

WORKDIR /opt/gurobi/linux64
#run the setup
RUN python setup.py install



FROM movesrwth/stormpy:1.7.0
# Mirror of the following Docker container
# FROM movesrwth/stormpy:ci-release
MAINTAINER Thom Badings <thom.badings@ru.nl>


# Build
#############
RUN mkdir /opt/sensitivity
WORKDIR /opt/sensitivity
# Obtain requirements and install them
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Only then install remainder
COPY . .

ENTRYPOINT /bin/bash
