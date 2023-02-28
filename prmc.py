from core.classes import PMC

from core.pmc_functions import pmc_load_instantiation, pmc_instantiate, pmc_get_reward
from core.export import export_json, timer

from core.prmc_functions import pmc2prmc, prmc_verify, prmc_derivative_LP, prmc_validate_derivative

from core.parser import parse_main

import os
from pathlib import Path
from tabulate import tabulate
from datetime import datetime

import numpy as np

# Parse arguments
args = parse_main()

# Load PRISM model with STORM
args.root_dir = os.path.dirname(os.path.abspath(__file__))

# args.model = 'models/slipgrid/double_pmc_size=10_params=10_seed=0.drn'
# args.parameters = 'models/slipgrid/double_pmc_size=10_params=10_seed=0_mle.json'
# args.formula = 'Rmin=? [F "goal"]'

# args.num_deriv = 10

args.model = 'models/pdtmc/brp64_4.pm'
args.formula = 'P=? [ F s=5 ]'
args.default_valuation = 0.9
args.goal_label = '(s = 5)'

# args.model = 'models/pmdp/CSMA/csma2_4_param.nm'
# args.formula = 'R{"time"}max=? [ F "all_delivered" ]'
# args.default_valuation = 0.1
# args.goal_label = 'all_delivered'

T = timer()

### pMC execution

from gurobipy import GRB
args.derivative_direction = GRB.MAXIMIZE

current_time = datetime.now().strftime("%H:%M:%S")
print('Program started at {}'.format(current_time))
print('\n',tabulate(vars(args).items(), headers=["Argument", "Value"]),'\n')

model_path = Path(args.root_dir, args.model)
param_path = Path(args.root_dir, args.parameters) if args.parameters else False

# Define pMC
pmc = PMC(model_path = model_path, args = args)
    
# Obtain parameter instantiation from file
inst = pmc_load_instantiation(pmc, args, param_path)

# Define instantiated pMC based on parameter valuation
instantiated_model, inst['point'] = pmc_instantiate(pmc, inst['valuation'], T)
pmc.reward = pmc_get_reward(pmc, args, inst)

#####
#####

if inst['sample_size'] is None:
    print('- Create arbitrary sample sizes')
    inst['sample_size'] = {par.name: 1000 + np.random.rand()*10 for par in pmc.parameters}

# scale rewards for prMC:
if args.scale_reward:
    pmc.reward = pmc.reward / np.max(pmc.reward)

print('Start prMC code')

args.uncertainty_model = "Hoeffding"
args.robust_probabilities = np.full(pmc.model.nr_states, True)
args.robust_dependencies = 'none' # Can be 'none' or 'parameter'
args.default_sample_size = 1000

print('Convert pMC to prMC...')
prmc = pmc2prmc(pmc.model, pmc.parameters, inst['point'], inst['sample_size'], args, args.verbose, T)

print('Verify prMC...')
P, solution = prmc_verify(prmc, pmc, args, args.verbose, T)

print('\nSet up sensitivity computation...')
deriv = prmc_derivative_LP(prmc, pmc, P, args, T)

# assert False

if not args.no_gradient_validation:
    deriv = prmc_validate_derivative(prmc, pmc, inst, solution, deriv, args)

if not args.no_export:
    export_json(args, prmc, T, inst, solution, deriv, parameters = pmc.parameters)

current_time = datetime.now().strftime("%H:%M:%S")
print('\nprMC code ended at {}\n'.format(current_time))
print('=============================================')