from core.classes import PMC

from core.pmc_functions import pmc_load_instantiation, pmc_instantiate, pmc_derivative_LP, pmc_validate_derivative, assert_probabilities
from core.verify_pmc import pmc_verify, pmc_get_reward
from core.export import export_json, timer

from core.parser import parse_main

import os
from pathlib import Path
from tabulate import tabulate
from datetime import datetime

# Parse arguments
args = parse_main()

# args.model = '/home/thom/Documents/CAV23/code/models/slipgrid/double_pmc_size=200_params=1000_seed=0.drn'
# args.parameters = '/home/thom/Documents/CAV23/code/models/slipgrid/double_pmc_size=200_params=1000_seed=0_mle.json'
# args.formula = 'Rmin=? [F "goal"]'
# args.pMC_engine = 'spsolve'
# args.output_folder = 'output/slipgrid__2023_03_09_11_31_26/'
# args.num_deriv = 10
# args.explicit_baseline = True
# args.robust_bound = 'lower'
# args.scale_reward = True
# args.no_gradient_validation = True

# args.model = 'models/sttt-drone/drone_model.nm'
# args.formula = 'Pmax=? [F attarget ]'
# args.default_valuation = 1/13
# args.goal_label = {'(((x > (15 - 2)) & (y > (15 - 2))) & (z > (15 - 2)))'}

# args.model = 'models/pdtmc/brp16_2.pm'
# args.formula = 'P=? [ F s=5 ]'
# args.default_valuation = 0.9
# args.goal_label = {'(s = 5)'}

# args.model = 'models/pdtmc/nand5_10.pm'
# args.formula = 'P=? [F \"target\" ]'
# args.parameters = 'models/pdtmc/nand.json'
# args.goal_label = {'target'}

# args.model = 'models/pmdp/wlan/wlan0_param.nm'
# args.formula = 'R{"time"}max=? [ F s1=12 | s2=12 ]'
# args.default_valuation = 0.01
# args.robust_bound = 'upper'

# Load PRISM model with STORM
args.root_dir = os.path.dirname(os.path.abspath(__file__))

args.beta_penalty = 0
args.validate_delta = 1e-5

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
inst = pmc_load_instantiation(pmc, param_path, args.default_valuation)

# Define instantiated pMC based on parameter valuation
instantiated_model, inst['point'] = pmc_instantiate(pmc, inst['valuation'], T)
assert_probabilities(instantiated_model)

pmc.reward = pmc_get_reward(pmc, instantiated_model, args)

print('\n',instantiated_model,'\n')

# Verify pMC
solution, J, Ju = pmc_verify(instantiated_model, pmc, inst['point'], T)

deriv = pmc_derivative_LP(pmc, J, Ju, args, T)

if not args.no_gradient_validation:
    pmc_validate_derivative(pmc, inst, solution, deriv, args.validate_delta)
    
if not args.no_export:
    export_json(args, pmc, T, inst, solution, deriv)

current_time = datetime.now().strftime("%H:%M:%S")
print('\npMC code ended at {}\n'.format(current_time))
print('=============================================')