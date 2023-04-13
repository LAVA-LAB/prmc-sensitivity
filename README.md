# Efficient Sensitivity Analysis for Parametric Robust Markov Chains

This is an implementation of the approach proposed in the paper:

- [1] "Efficient Sensitivity Analysis for Parametric Robust Markov Chains" by Thom Badings, Sebastian Junges, Ahmadreza Marandi, Ufuk Topcu, and Nils Jansen, CAV 2023

The methods in this artifact can be used to compute the partial derivatives of the solution functions for parametric and parametric robust Markov chains.
More specifically, we compute the partial derivatives of the solution function with respect to the parameters of these Markov models.
The artifact also contains an implementation of these methods in a learning framework, in which these derivatives are used to guide the exploration process (i.e., determining where to sample).

## 1. Run from a Docker container

We provide a docker container. We assume you have Docker installed (if not, see the [Docker installation guide](https://docs.docker.com/get-docker/)). Then, run:

```
docker pull thombadings/sensitivity:cav23
```

or in case you downloaded this container from an (unpacked) archive:

```
docker load -i sensitivity_cav23_docker.tar
```

Our Docker container is built upon containers for [Gurobi Optimization](https://hub.docker.com/r/gurobi/optimizer) and for the [probabilistic model checker Storm](https://www.stormchecker.org/documentation/obtain-storm/docker.html) (click the links to the documentation for details).

Our implementation uses Gurobi to solve linear programs. There are two options for running the container. Although you can solve optimization problems of limited size without a Gurobi license, a license is required to run our bigger benchmarks. Luckily, Gurobi offers free academic licenses. To obtain such a license, you can [follow the steps on this page](https://www.gurobi.com/features/academic-wls-license/). 

Important: Make sure to obtain an Academic Web License Service (WLS) License! Other options, such as a named-user license will not work in combination with the Docker container.

My important paragraph.
{: .alert .alert-info}


To use the docker container, open a terminal and navigate to the folder which you want to use to synchronize results.


Moreover, 

--mount type=bind,source="$(pwd)",target=/opt/slurf/output -w /opt/slurf


 Then, run the container using the following command:

```
sudo docker run --env=GRB_CLIENT_LOG=3 --volume={PATH_TO_GUROBI_LICENSE_FILE}:/opt/gurobi/gurobi.lic:ro --mount type=bind,source="$(pwd)",target=/opt/sensitivity/output -it cav23


docker run  --rm -it --name slurf thombadings/slurf:cav22
```

You will see a prompt inside the docker container. The README in this folder is what you are reading. Now you are ready to run the code for a single model (Section 3) or to replicate the experiments presented in [1] (Section 4).




## 1. Installation from source


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
