# Efficient Sensitivity Analysis for Parametric Robust Markov Chains

This is an implementation of the approach proposed in the paper:

- [1] "Efficient Sensitivity Analysis for Parametric Robust Markov Chains" by Thom Badings, Sebastian Junges, Ahmadreza Marandi, Ufuk Topcu, and Nils Jansen, CAV 2023

The methods in this artifact can be used to compute the partial derivatives of the solution functions for parametric Markov chains (pMCs) and parametric robust Markov chains (prMCs).
More specifically, we compute the partial derivatives of the solution function with respect to the parameters of these Markov models.
The artifact also contains an implementation of these methods in a learning framework, in which these derivatives are used to guide the exploration process (i.e., determining where to sample).

# 1. Run from a Docker container

We provide a docker container. To use the container, you can follow the steps below.

### Step 1: Pull or download the Docker container
We assume you have Docker installed (if not, see the [Docker installation guide](https://docs.docker.com/get-docker/)). Then, run:

```
docker pull thombadings/sensitivity:cav23
```

or in case you downloaded this container from an (unpacked) archive:

```
docker load -i sensitivity_cav23_docker.tar
```

Our Docker container is built upon containers for [Gurobi Optimization](https://hub.docker.com/r/gurobi/optimizer) and for the [probabilistic model checker Storm](https://www.stormchecker.org/documentation/obtain-storm/docker.html) (click the links to the documentation for details).

### Step 2: Obtain a WLS Gurobi license
Gurobi is used to solve linear programs. Although you can solve optimization problems of limited size without a Gurobi license, a license is required to run our bigger benchmarks. Luckily, Gurobi offers free academic licenses. To obtain such a license, you can [follow the steps on this page](https://www.gurobi.com/features/academic-wls-license/). 

{-Important: Make sure to obtain an Academic Web License Service (WLS) License! Other options, such as a named-user license will not work in combination with the Docker container.-}

After obtaining the license, download the license file (`Gurobi.lic`) and store it somewhere on your computer. To use the docker container, open a terminal and navigate to the folder which you want to use to synchronize results.

### Step 3: Run the Docker container
Then, run the following command, where you replace `{PATH_TO_GUROBI_LICENSE_FILE}` by the path to the `Gurobi.lic` file:

```
sudo docker run --env=GRB_CLIENT_LOG=3 --volume={PATH_TO_GUROBI_LICENSE_FILE}:/opt/gurobi/gurobi.lic:ro --mount type=bind,source="$(pwd)",target=/opt/sensitivity/output -it cav23
```

You will see a prompt inside the docker container. The README in this folder is what you are reading. Now you are ready to run the code for a single model (Section 3) or to replicate the experiments presented in [1] (Section 4).

# 2. Installation from source

While for users, we recommend to use the Docker container, you can also build our tool from source as follows:

- Install [Storm](https://www.stormchecker.org/documentation/obtain-storm/build.html), [pycarl](https://moves-rwth.github.io/pycarl/installation.html#installation-steps) and [stormpy](https://moves-rwth.github.io/stormpy/installation.html#installation-steps) using the instructions in the stormpy documentation.

  Note that one must use at least version 1.7.
  Preferably, install pycarl and stormpy in a virtual environment.

- Obtain a Gurobi license and activate it on your machine.

- Install the Python dependencies with:

  `pip install -r requirements.txt`

# 3. Running for a single model

There are three Python files that can be run:

1. `run_pmc.py`
2. `run_prmc.py`
3. `run_learning.py`

The `run_pmc.py` and `run_prmc.py` files compute the partial derivatives for a pMC and prMC, respectively, while the `run_learning.py` file runs the learning experiment.

A miminal command to compute derivatives for a pMC is as follows:

```
python run_pmc.py --model <path to model file> --parameters <path to parameters file> --formula <formula to check> --num_deriv <number of derivatives to compute>
```

For example, to compute the `k=10` highest derivatives for pMC of the 50x50 slippery grid world benchmark with 100 parameters, you run:

```
timeout 3600s python3 run_pmc.py --instance "grid(50,100,double)" --model 'models/slipgrid/double_pmc_size=50_params=100_seed=0.drn' --parameters 'models/slipgrid/double_pmc_size=50_params=100_seed=0_mle.json' --formula 'Rmin=? [F "goal"]' --num_deriv 10;
```

The command computes the solution of the given formula for this pMC (see Section 6 for details on the input model format), given the parameter instantiation provided in the JSON file, and computes the 10 highest partial derivatives.
The results are then saved to a JSON file in the `output/` folder.

The equivalent command to compute derivatives for the corresponding prMC is:

```
timeout 3600s python3 run_prmc.py --instance "grid(50,100,double)" --model 'models/slipgrid/double_pmc_size=50_params=100_seed=0.drn' --parameters 'models/slipgrid/double_pmc_size=50_params=100_seed=0_mle.json' --formula 'Rmin=? [F "goal"]' --num_deriv 10;
```

There is a variety of arguments that you can add to these scripts, in order to further customize the execution. See Section 5 for a complete overview of all available arguments.

# 4. Reproducing results in the paper

You can reproduce the figures and tables presented in our paper [1] by following the steps below.
Before running the experiments, we recommend to remove any existing files/folders in the output/ folder (except the .keep file).

The experiments consist of three parts:

### Part 1: Derivatives for slippery grid worlds
The following comand runs all experiments on the slippery grid worlds:

```
cd experiments; bash grid_world.sh
```

The results are exported in a CSV file `output/slipgrid_table.csv` and LaTeX table `output/slipgrid_table.tex`.
Tables 1 and 2 of the paper [1] are directly obtained through this LaTeX table.

### Part 2: Derivatives for benchmarks from literature
The following comand runs all other benchmarks used in [1]:

```
cd experiments; bash benchmarks_cav23.sh
```

Each benchmark has a timeout of 1 hour, so running al benchmarks can take long.

For faster evaluation, you can also run a reduced set of benchmarks using:

```
cd experiments; bash benchmarks_cav23_partial.sh
```

The results are exported to a CSV file `output/benchmarks_cav23.csv` and a LaTeX table `output/benchmarks_cav23.tex`.
Table 3 is the paper [1] is directly obtained through this LaTeX table.

### Part 3: Learning framework
The following Python script reproduces the application in a learning framework, presented in Section 6, Q3 of [1]:

```
python3 run_learning.py
```

This script runs the learning framework experiment for both the 20x20 slippy grid world, as well as the drone navigation problem, exactly as presented in [1].
The script creates a plot `learning_{model}_{datetime}.pdf`, where `model` is either `slipgrid` or `drone`, and `{datetime}` is the datetime at which the file was generated.
Moreover, Figure 7 of [1] is generated for both models using the CSV files `learning_{model}_{datetime}.csv`.

### Recreating experiment shell scripts
If you wish to change these experiments or reproduce these scripts (and the corresponding model), run the `generate_experiments.py` file in the root of this repository.
This script (re)creates the shell scripts in the `experiments/` folder, as well as the corresponding pMC models (e.g., the randomized slippy grid worlds) in the `models/` folder.

# 5. Overview of available arguments

# 6. Defining parametric (robust) Markov chains

# 7. Rebuilding the Docker container

The included Docker image of our artifact is based on Docker images of [Gurobi](https://hub.docker.com/r/gurobi/optimizer) and [Stormpy](https://www.stormchecker.org/documentation/obtain-storm/docker.html). After making changing to the source code, the Docker container must be built again using the included Dockerfile. Rebuilding the image can be done by executing the following command in the root directory of the artiact (here, 1.0 indicates the version):

```
sudo docker build -t sensitivity:1.0 .
```
