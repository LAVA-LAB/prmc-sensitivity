from core.classes import PMC

from core.pmc_functions import pmc_load_instantiation, pmc_instantiate, assert_probabilities
from core.verify_pmc import pmc_get_reward
from core.baseline_gradient import explicit_gradient
from core.io.export import export_json, timer
from core.prmc_functions import pmc2prmc, prmc_verify, prmc_derivative_LP, prmc_validate_derivative
from core.io.parser import parse_main

from pathlib import Path
from tabulate import tabulate
from datetime import datetime

import os
import time
import numpy as np

# Parse arguments
args = parse_main()

# Load PRISM model with STORM
args.root_dir = os.path.dirname(os.path.abspath(__file__))

T = timer()

### pMC execution

from gurobipy import GRB
args.derivative_direction = GRB.MAXIMIZE

current_time = datetime.now().strftime("%H:%M:%S")
print('Program started at {}'.format(current_time))
print('\n',tabulate(vars(args).items(), headers=["Argument", "Value"]),'\n')

start_time = time.time()

model_path = Path(args.root_dir, args.model)
param_path = Path(args.root_dir, args.parameters) if args.parameters else False

# Define pMC
pmc = PMC(model_path = model_path, args = args)
    
# Obtain parameter instantiation from file
inst = pmc_load_instantiation(pmc, param_path, args.default_valuation)

# Define instantiated pMC based on parameter valuation
instantiated_model, inst['point'] = pmc_instantiate(pmc, inst['valuation'], T)
assert_probabilities(instantiated_model)

pmc.reward = pmc_get_reward(pmc, instantiated_model, args)

print('\n',instantiated_model,'\n')

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
args.default_sample_size = 1000

print('Convert pMC to prMC...')
prmc = pmc2prmc(pmc.model, pmc.parameters, pmc.scheduler_prob, inst['point'], inst['sample_size'], args, args.verbose, T)

T.times['initialize_model'] = time.time() - start_time

print('Verify prMC...')
P, solution = prmc_verify(prmc, pmc, args, args.verbose, T)

print('\nSet up sensitivity computation...')
G, deriv = prmc_derivative_LP(prmc, pmc, P, args, T)

if args.explicit_baseline:
    print('-- Execute baseline: compute all gradients explicitly')
    deriv['explicit'] = explicit_gradient(prmc, args, G.J, G.Ju, T)

if not args.no_gradient_validation:
    deriv = prmc_validate_derivative(prmc, pmc, inst, solution, deriv, 
                     args.validate_delta, args.robust_bound, args.beta_penalty)

if not args.no_export:
    export_json(args, prmc, T, inst, solution, deriv)

current_time = datetime.now().strftime("%H:%M:%S")
print('\nprMC code ended at {}\n'.format(current_time))
print('=============================================')