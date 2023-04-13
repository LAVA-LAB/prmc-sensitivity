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

Gurobi is used to solve linear programs. Although you can solve optimization problems of limited size without a Gurobi license, a license is required to run our bigger benchmarks. Luckily, Gurobi offers free academic licenses. To obtain such a license, you can [follow the steps on this page](https://www.gurobi.com/features/academic-wls-license/). 

- {-Important: Make sure to obtain an Academic Web License Service (WLS) License! Other options, such as a named-user license will not work in combination with the Docker container.-}

After obtaining the license, download the license file (`Gurobi.lic`) and store it somewhere on your computer. To use the docker container, open a terminal and navigate to the folder which you want to use to synchronize results.

Then, run the following command, where you replace `{PATH_TO_GUROBI_LICENSE_FILE}` by the path to the `Gurobi.lic` file:

```
sudo docker run --env=GRB_CLIENT_LOG=3 --volume={PATH_TO_GUROBI_LICENSE_FILE}:/opt/gurobi/gurobi.lic:ro --mount type=bind,source="$(pwd)",target=/opt/sensitivity/output -it cav23
```

You will see a prompt inside the docker container. The README in this folder is what you are reading. Now you are ready to run the code for a single model (Section 3) or to replicate the experiments presented in [1] (Section 4).

## 2. Installation from source

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

## 3. Running for a single model

There are three Python files that can be run:

1. `run_pmc.py`
2. `run_prmc.py`
3. `run_learning.py`

The `run_pmc.py` and `run_prmc.py` files compute the partial derivatives for a pMC and prMC, respectively, while the `run_learning.py` file runs the learning experiment.

A miminal command to compute derivatives for a pMC is as follows:

```
python run_pmc.py --model <path to model file> --parameters <path to parameters file> --formula <formula to check> --num_deriv <number of derivatives to compute>
```

<Explain the arguments...>

For example, to compute the `k=10` highest derivatives for pMC of the 50x50 slippery grid world benchmark with 100 parameters, you run:

```
timeout 3600s python3 run_pmc.py --instance "grid(50,100,double)" --model 'models/slipgrid/double_pmc_size=50_params=100_seed=0.drn' --parameters 'models/slipgrid/double_pmc_size=50_params=100_seed=0_mle.json' --formula 'Rmin=? [F "goal"]' --num_deriv 10;
```

The equivalent command to run for the corresponding prMC is:

```
timeout 3600s python3 run_prmc.py --instance "grid(50,100,double)" --model 'models/slipgrid/double_pmc_size=50_params=100_seed=0.drn' --parameters 'models/slipgrid/double_pmc_size=50_params=100_seed=0_mle.json' --formula 'Rmin=? [F "goal"]' --num_deriv 10;
```

There is a variety of arguments that you can add to these scripts, in order to further customize the execution. See Section 5 for a complete overview of all available arguments.

## 4. Reproducing results in the paper

## 5. Overview of available arguments

## 6. Defining parametric (robust) Markov chains

## 7. Rebuilding the Docker container

The included Docker image of our artifact is based on Docker images of [Gurobi](https://hub.docker.com/r/gurobi/optimizer) and [Stormpy](https://www.stormchecker.org/documentation/obtain-storm/docker.html). After making changing to the source code, the Docker container must be built again using the included Dockerfile. Rebuilding the image can be done by executing the following command in the root directory of the artiact (here, 1.0 indicates the version):

```
sudo docker build -t sensitivity:1.0 .
```
