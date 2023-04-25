# Docker file for the sensitivity analysis program.
# This container builds upon stormpy, and then adds
# gurobi on top of this.

FROM movesrwth/stormpy:1.7.0
MAINTAINER Thom Badings <thom.badings@ru.nl>

ARG GRB_VERSION=10.0.0
ARG GRB_SHORT_VERSION=10.0

# install gurobi package and copy the files
WORKDIR /opt

RUN apt-get update \
    && apt-get install --no-install-recommends -y\
       ca-certificates  \
       p7zip-full \
       zip \
       wget \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates \
    && wget -v https://packages.gurobi.com/${GRB_SHORT_VERSION}/gurobi${GRB_VERSION}_linux64.tar.gz \
    && tar -xvf gurobi${GRB_VERSION}_linux64.tar.gz  \
    && rm -f gurobi${GRB_VERSION}_linux64.tar.gz \
    && mv -f gurobi* gurobi \
    && rm -rf gurobi/linux64/docs

ENV GUROBI_HOME /opt/gurobi/linux64
ENV PATH $PATH:$GUROBI_HOME/bin
ENV LD_LIBRARY_PATH $GUROBI_HOME/lib

WORKDIR /opt/gurobi/linux64
#run the setup
RUN python setup.py install

# Build artifact dependencies
#############
RUN mkdir /opt/sensitivity
WORKDIR /opt/sensitivity
# Obtain requirements and install them
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Only then install remainder
COPY . .

ENTRYPOINT /bin/bash
