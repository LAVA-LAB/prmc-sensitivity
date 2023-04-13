# Efficient Sensitivity Analysis for Parametric Robust Markov Chains

This is an implementation of the approach proposed in the paper:

- [1] "Efficient Sensitivity Analysis for Parametric Robust Markov Chains" by Thom Badings, Sebastian Junges, Ahmadreza Marandi, Ufuk Topcu, and Nils Jansen, CAV 2023

1. Install Storm, pycarl, and stormpy
2. Install packages with: 
    $ pip install -r requirements.txt
3. Install scikit-umfpack:
    $ git clone https://github.com/scikit-umfpack/scikit-umfpack.git
    $ cd scikit-umfpack
    $ python setup.py install

Activate environment:
screen -S CAV23;
source prmdps/bin/activate;


Build Docker container



Run Docker container

sudo docker run --env=GRB_CLIENT_LOG=3 --volume=/home/thom/gurobi_docker.lic:/opt/gurobi/gurobi.lic:ro -it cav23
