Efficient Sensitivity Analysis for Parametric Robust Markov Chains
=============================

This repository contains a Python implementation of the approach proposed in the paper:

- [1] "Efficient Sensitivity Analysis for Parametric Robust Markov Chains" by Thom Badings, Sebastian Junges, Ahmadreza Marandi, Ufuk Topcu, and Nils Jansen (CAV 2023)

The methods in this artifact can be used to compute the partial derivatives of the solution functions for parametric Markov chains (pMCs) and parametric robust Markov chains (prMCs).
More specifically, we compute the $k$ highest (or lowest) partial derivatives of the solution function with respect to the parameters of these Markov models.
The artifact also contains an implementation of these methods in a learning framework, in which these derivatives are used to guide the exploration process (i.e., determining where to sample).

Contents of this ReadMe:
1. [Run from a Docker container (preferred)](#1-run-from-a-docker-container-preferred)
2. [Installation from source](#2-installation-from-source)
3. [Running for a single model](#3-running-for-a-single-model)
4. [Reproducing results in the paper](#4-reproducing-results-in-the-paper)
5. [Overview of available arguments](#5-overview-of-available-arguments)
6. [Defining parametric (robust) Markov chains](#6-defining-parametric-robust-markov-chains)
7. [Rebuilding the Docker container](#7-rebuilding-the-docker-container)

We have tested the artifact using Python 3.10, Storm/Stormpy 1.7, Gurobi/Gurobipy 10.0.0, and Docker 23.0.4. The source code is available on GitHub via [https://github.com/LAVA-LAB/prmc-sensitivity](https://github.com/LAVA-LAB/prmc-sensitivity).

# 1. Run from a Docker container (preferred)

The preferred way to run our code is using the Docker container that we provide.

> **_NOTE:_** On Ubuntu, Docker binds by default to a socket that other users can only access using sudo. Thus, it may be necessary to run Docker commands using sudo. Alternatively, one can follow [this guide on the Docker website](https://docs.docker.com/engine/install/linux-postinstall/) to avoid having to run with sudo.

### Step 1: Pull or download the Docker container
We assume you have Docker installed (if not, see the [Docker installation guide](https://docs.docker.com/get-docker/)). Then, run:

```
docker pull thombadings/prmc_sensitivity:cav23
```

Or in case you downloaded this container from an (unpacked) archive (loading the container an take a few minutes):

```
docker load -i prmc_sensitivity_cav23_docker.tar
```

Our Docker container is built upon a container for the [probabilistic model checker Storm](https://www.stormchecker.org/documentation/obtain-storm/docker.html), on top of which we build Gurobi and install the Python dependencies (see the Dockerfile in this artifact for details).

### Step 2: Obtain a WLS Gurobi license
Gurobi is used to solve linear programs. Although you can solve optimization problems of limited size without a Gurobi license, a license is required to run our bigger benchmarks. Luckily, Gurobi offers free academic licenses. To obtain such a license, [follow the steps on this page](https://www.gurobi.com/features/academic-wls-license/). 

> **_NOTE:_**  Make sure to obtain an Academic Web License Service (WLS) License! Other options, such as a named-user license, will not work in combination with the Docker container.

After obtaining the license, download the license file (`Gurobi.lic`) and store it somewhere on your computer.

### Step 3: Run the Docker container
To use the docker container, open a terminal and navigate to the folder which you want to use to synchronize results.
Then, run the following command, where you replace `{PATH_TO_GUROBI_LICENSE_FILE}` by the path to the `Gurobi.lic` WLS license file, for example `/home/thom/gurobi_docker.lic`:

```
docker run --env=GRB_CLIENT_LOG=3 --volume={PATH_TO_GUROBI_LICENSE_FILE}:/opt/gurobi/gurobi.lic:ro --mount type=bind,source="$(pwd)",target=/opt/sensitivity/output -it thombadings/prmc_sensitivity:cav23
```

You will see a prompt inside the docker container. The README in this folder is what you are reading. Now you are ready to run the code for a single model (Section 3) or to replicate the experiments presented in [1] (Section 4).

# 2. Installation from source

While for users, we recommend using the Docker container, you can also build our tool from source as follows:

1. Install [Storm](https://www.stormchecker.org/documentation/obtain-storm/build.html), [Pycarl](https://moves-rwth.github.io/pycarl/installation.html#installation-steps) and [Stormpy](https://moves-rwth.github.io/stormpy/installation.html#installation-steps) using the instructions in the stormpy documentation. We have tested the artifact using Storm and Stormpy version 1.7. Preferably, install pycarl and stormpy in a virtual environment.

2. Install [Gurobi](https://www.gurobi.com/downloads/), and obtain a Gurobi license and activate it on your machine. For example, for an [(academic) named-user license](https://www.gurobi.com/features/academic-named-user-license/), run the `grbgetkey` command with your obtained license key (recall that for usage with Docker, you need a WLS license). We have tested the artifact using Gurobi and Gurobipy 10.0.0.

3. Install the Python dependencies with:

  ```
  pip install -r requirements.txt
  ```

# 3. Running for a single model

There are three Python files that can be run:

1. `run_pmc.py` - Compute partial derivatives for a pMC
2. `run_prmc.py` - Compute partial derivatives for a prMC
3. `run_learning.py` - Using derivatives to guide sampling in a learning framework

A minimal command to compute derivatives for a pMC is as follows:

```
python run_pmc.py --model <path-to-model-file> --parameters <path-to-parameters-file> --formula <formula-to-check> --num_deriv <number-of-derivatives-to-compute>
```

For example, to compute the $k=10$ highest derivatives for the pMC of the 50x50 slippery grid world benchmark (see [1] for details) with $|V|=100$ parameters, you run:

```
python3 run_pmc.py --instance "grid(50,100,double)" --model 'models/slipgrid/pmc_size=50_params=100.drn' --parameters 'models/slipgrid/pmc_size=50_params=100_mle.json' --formula 'Rmin=? [F "goal"]' --num_deriv 10;
```

The command first computes the solution of the given formula for this pMC (see Section 6 for details on the input model format), given the parameter instantiation provided in the JSON file.
Thereafter, the $k=10$ highest partial derivatives are computed.
The results are then saved to a JSON file in the `output/` folder.

The equivalent command to compute derivatives for the corresponding prMC is:

```
python3 run_prmc.py --instance "grid(50,100,double)" --model 'models/slipgrid/pmc_size=50_params=100.drn' --parameters 'models/slipgrid/pmc_size=50_params=100_mle.json' --formula 'Rmin=? [F "goal"]' --num_deriv 10;
```

An example to learn the learning framework for a 20x20 grid world with 100 terrain types (for all modeled exploration strategies, with 100 steps of obtaining 25 additional samples) is:

```
python3 run_learning.py --instance gridworld --model models/slipgrid_learning/pmc_size=20_params=100.drn --parameters models/slipgrid_learning/pmc_size=20_params=100_mle.json --formula 'Rmin=? [F "goal"]' --output_folder 'output/learning/' --num_deriv 1 --robust_bound 'upper' --uncertainty_model 'Hoeffding' --true_param_file models/slipgrid_learning/pmc_size=20_params=100.json --learning_iterations 1 --learning_steps 100 --learning_samples_per_step 25;
```

There are a variety of arguments that you can add to these scripts in order to customize the execution further. See Section 5 for a complete overview of all available arguments.

# 4. Reproducing results in the paper

You can reproduce the results presented in [1] by following the three steps below.

### Step 1: Creating experiment shell scripts
Before running the experiments, we recommend removing any existing result files/folders in the output/ folder.
Then, run the following command to (re)create the shell scripts in the `experiments/` folder, as well as the corresponding models (e.g., the randomized slippy grid worlds) in the `models/` folder:

```
python3 generate_experiments.py
```

### Step 2: Running experiments
Then, to reproduce the figures and tables presented in our paper [1], execute one of the shell scripts in the `experiments/` folder:

- `cd experiments; bash all_experiments_full.sh` runs the full set of experiments as presented in [1]. 
    * Expected run time: 24 hours for the grid world and standard benchmarks, plus 48 hours for the learning applications
    * Resource requirements: Tested using a computer with a 4GHz Intel Core i9 CPU and 64 GB RAM (only Gurobi running multi-threaded)
- `cd experiments; bash all_experiments_partial.sh` runs a partial set of experiments. 
    * Expected run time: 1 hour
    * Resource requirements: Tested using a computer with a 1.3GHz Intel Core i7 CPU and 16 GB RAM (only Gurobi running multi-threaded)

Both shell scripts, in turn, run three different sets of experiments, which can also be run independently from each other:

1. Computing derivatives on a variety of slippery grid world problems (`experiments/grid_world.sh` or `experiments/grid_world_partial.sh`).
2. Computing derivatives on a set of benchmarks from the literature (`experiments/benchmarks.sh` or `experiments/benchmarks_partial.sh`).
3. An application of our method in a learning framework on two different models (`run_learning.py`).

### Step 3: Recreating figures and tables
After running the experiments, the figures and tables presented in [1] can be reproduced as follows:

- Tables 1 and 2 (results for the grid world experiments) are obtained through the LaTeX table exported to `output/slipgrid_table.tex` (or `output/slipgrid_partial_table.tex`). This data is also exported to a CSV file with the same name.

- Table 3 (results for the benchmarks from the literature) is obtained through the LaTeX table exported to `output/benchmarks_table.tex` (or `output/benchmarks_partial_table.tex`). This data is also exported to a CSV file with the same name.

- Figure 5 (scatter plot for times to compute derivatives vs. solutions) is obtained using the data in `output/scatter_time_verify_vs_differentiating.csv`. This CSV file contains the model type, verification time (Verify), time to compute one derivative (OneDeriv), the time to compute the highest derivative (Highest), and the number of parameters of each instance (Parameters).

- Figure 6 (scatter plot for times to compute $k$ derivatives vs. all derivatives) is obtained using the data in `output/scatter_time_highest1_vs_all.csv` (left subfigure, for $k=1$) and `output/scatter_time_highest10_vs_all.csv` (right subfigure, for $k=10$). This CSV file contains the model type, time to compute the $k$ highest derivatives (Highest), time to compute all derivatives (All), and the number of parameters of each instance (Parameters). The time to compute all derivatives is extrapolated from the time to compute 10 derivatives explicitly.

- Figure 7 (results for the learning framework) is obtained using the data in `output/learning_gridworld_{datetime}.csv` and `output/learning_drone_{datetime}.csv`, where `{datetime}` is a datetime stamp of when the file is created. A Python version of the plots in Figure 7 is exported to `output/learning_gridworld_{datetime}.pdf` and `output/learning_drone_{datetime}.pdf`.

# 5. Overview of available arguments

Below, we list all arguments that can be passed to the commands for running the Python scripts. The following arguments are given as `--<argument name> <value>`: 

| Argument    | Required? | Default          | Type                     | Description |
| ---         | ---       | ---              | ---                      | ---         |
| model       | Yes       | n/a              | string                   | Model file, e.g., `models/slipgrid/pmc_size=50_params=100.drn` |
| formula     | Yes       | n/a              | string                   | Formula to verify |
| goal_label  | No        | None             | string                   | If the provided formula computes a reachability probability, the goal labels are those labels that correspond to goal states. Multiple goal labels can be passed, e.g., as `--goal_label "{'goal','notbad'}"`. |
| instance    | No        | False            | string                   | Name of the instance to run (used as tag in the result export files) |
| parameters  | No        | False            | string                   | Path to a parameter valuation file in JSON format (see Section 6 for details). If no file is provided, the `--default_valuation` argument is used. |
| default_valuation | No  | 0.5              | float                    | Default parameter valuation. This value is assigned to every parameter, unless the `parameters` argument is provided. |
| num_deriv   | No        | 1                | int                      | Number of derivatives to compute |
| uncertainty_model | No  | Linf             | string                   | The type of uncertainty model used for prMCs, which can be `Linf` (infinity norm), `L1` (1-norm), or `Hoeffding` (using Hoeffding's inequality to obtain probability intervals) |
| discount    | No        | 1                | float in [0,1]           | Discount factor |
| output_folder | No      | output/          | string                   | Folder in which to export results |
| validate_delta | No     | 1e-4             | float                    | Perturbation value used to validate gradients (not used if `--no_gradient_validation` is also passed) |
| robust_bound | No       | upper            | string                   | Determines which robust bound to compute for prMCs. Can be either `upper` or `lower`. |
| robust_confidence | No  | 0.9              | float in [0,1]           | Confidence level on individual PAC probability intervals (only used if `Hoeffding` is used as value for `--uncertainty_model`) |

Moreover, a number of boolean arguments can be added as `--<argument name>`:

| Argument    | Required? | Default          | Type                     | Description |
| ---         | ---       | ---              | ---                      | ---         |
| no_par_dependencies | No | False           | boolean                  | If added, parameter dependencies between distributions are avoided for prMCs. |
| scale_reward | No       | False            | boolean                  | If added, rewards for prMCs are normalized to one (can improve numerical stability for high solutions) |
| verbose     | No        | False            | boolean                  | If added, additional output is provided by the scripts |
| no_export   | No        | False            | boolean                  | If added, no results are exported |
| no_gradient_validation | No | False            | boolean              | If added, the validation of computed derivatives by an empirical perturbation analysis is skipped |
| explicit_baseline | No  | False            | boolean                  | If added, a baseline that computes all partial derivatives explicitly is performed |

Finally, a number of arguments are only used by the learning framework (`run_learning.py`):

| Argument        | Required? | Default          | Type                     | Description |
| ---             | ---       | ---              | ---                      | ---         |
| true_param_file | No        | False            | string                   | Path to the file with the true parameter valuations. If not passed, the `--default_valuation` value is used for the true parameter values. |
| learning_iterations | No    | 1                | int                      | Number of iterations for each exploration method in the learning framework |
| learning_steps  | No        | 100              | int                      | Number of steps to take in each iteration |
| learning_samples_per_step | No | 25            | int                      | Number of additional samples to obtain in each step in the learning framework |
| default_sample_size | No    | 100              | int                      | Default number of samples for each parameter to start with (not used if these sample sizes are already provided in the file passed to `--parameters`) |

# 6. Defining parametric (robust) Markov chains

### Parametric Markov chains
Our implementation supports pMCs defined in [standard Prism format](https://prismmodelchecker.org/manual/ThePRISMLanguage/Introduction), or in explicit format (`*.drn` files).
Using explicit format can significantly reduce the time to parse large models, and is thus used for the grid world benchmarks.
See the .drn files in the `models/slipgrid/` folder for examples of how these models are defined.

### Parametric robust Markov chains
When running `run_prmc.py` for prMCs, the script actually loads a pMC and extends this model into a prMC, by creating uncertainty sets (of the type passed to the `--uncertainty_model` argument) around the provided parameter instantiation.

### Parameter valuation files
A parameter instantiation can be passed through the `--default_valuation` argument (see Section 5), or by providing a file using the `--parameters` argument.
This parameter instantiation file assigns a value to each parameter and (optionally) a sample size for each of the parameters.
For example, the slippery grid world model in `models/slipgrid/pmc_size=10_params=10.drn` has 10 parameters, so its parameter instantiation file `models/slipgrid/pmc_size=10_params=10_mle.json` is defined as:

```
{"v0": [0.20853080568720378, 844], "v1": [0.19893617021276597, 940], "v2": [0.15224191866527634, 959], "v3": [0.10526315789473684, 608], "v4": [0.13810741687979539, 782], "v5": [0.10300429184549356, 932], "v6": [0.13925729442970822, 754], "v7": [0.18058455114822547, 958], "v8": [0.12708333333333333, 960], "v9": [0.12384473197781885, 541]}
```

Here, each parameter is assigned a value and a sample size. For example, parameter v0 has a value of 0.2085 and a sample size of 844.
The sample sizes are used for prMCs with Hoeffding's inequality as uncertainty model.
For pMCs, it is also possible to simply omit the sample sizes, for example as in the file `models/slipgrid/pmc_size=10_params=10.json`:

```
{"v0": 0.1971945002499666, "v1": 0.1878193471347177, "v2": 0.15096243767199002, "v3": 0.10557146937016064, "v4": 0.14511592145209282, "v5": 0.10199876654087588, "v6": 0.14417109212488455, "v7": 0.19795867288127286, "v8": 0.13594444639693215, "v9": 0.1480893530836163}
```

# 7. Rebuilding the Docker container

The included Docker image of our artifact is based on Docker images of [Gurobi](https://hub.docker.com/r/gurobi/optimizer) and [Stormpy](https://www.stormchecker.org/documentation/obtain-storm/docker.html). After making changes to the source code, the Docker container must be built again using the included Dockerfile. Rebuilding the image can be done by executing the following command in the root directory of the artifact (here, 1.0 indicates the version):

```
docker build -t prmc_sensitivity:1.0 .
```

If Docker returns permission errors, consider running the command above with `sudo` (or see the note earlier in this ReadMe for avoid having to run using sudo).
