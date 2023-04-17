from core.classes import PMC

from core.pmc_functions import pmc_load_instantiation, pmc_instantiate, pmc_derivative_LP, pmc_validate_derivative, assert_probabilities
from core.baseline_gradient import explicit_gradient
from core.verify_pmc import pmc_verify, pmc_get_reward
from core.io.export import export_json, timer

from core.io.parser import parse_main

import os
import time

from pathlib import Path
from tabulate import tabulate
from datetime import datetime

# Parse arguments
args = parse_main()

# Load PRISM model with STORM
args.root_dir = os.path.dirname(os.path.abspath(__file__))
args.validate_delta = 1e-5

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

T.times['initialize_model'] = time.time() - start_time
print('\n',instantiated_model,'\n')

print('Verify pMC...')
solution, J, Ju = pmc_verify(instantiated_model, pmc, inst['point'], T)

print('\nSet up sensitivity computation...')
optm, deriv = pmc_derivative_LP(pmc, J, Ju, args, T)

if args.explicit_baseline:
    print('-- Execute baseline: compute all gradients explicitly')
    deriv['explicit'] = explicit_gradient(pmc, args, J, Ju, T)

if not args.no_gradient_validation:
    pmc_validate_derivative(pmc, inst, solution, deriv, args.validate_delta)
    
if not args.no_export:
    export_json(args, pmc, T, inst, solution, deriv)

current_time = datetime.now().strftime("%H:%M:%S")
print('\npMC code ended at {}\n'.format(current_time))
print('=============================================')