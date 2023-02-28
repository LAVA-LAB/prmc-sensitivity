from core.classes import PMC

from core.pmc_functions import pmc_load_instantiation, pmc_instantiate, pmc_get_reward, pmc_verify, pmc_derivative_LP, pmc_validate_derivative
from core.export import export_json, timer

from core.parser import parse_main

import os
from pathlib import Path
from tabulate import tabulate
from datetime import datetime

# Parse arguments
args = parse_main()

# Load PRISM model with STORM
args.root_dir = os.path.dirname(os.path.abspath(__file__))

args.model = 'models/slipgrid/double_pmc_size=10_params=10_seed=0.drn'
args.parameters = 'models/slipgrid/double_pmc_size=10_params=10_seed=0_mle.json'
args.formula = 'Rmin=? [F "goal"]'

args.num_deriv = 10

# args.model = 'models/pdtmc/brp64_4.pm'
# args.formula = 'P=? [ F s=5 ]'
# args.default_valuation = 0.9
# args.goal_label = '(s = 5)'

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
    
# Obtain parameter instantiatoin from file
inst = pmc_load_instantiation(pmc, args, param_path)

# Define instantiated pMC based on parameter valuation
instantiated_model, inst['point'] = pmc_instantiate(pmc, inst['valuation'], T)
pmc.reward = pmc_get_reward(pmc, args, inst)

solution, J, Ju = pmc_verify(instantiated_model, pmc, inst['point'], args, T)

deriv = pmc_derivative_LP(pmc, J, Ju, args, T)

if not args.no_gradient_validation:
    pmc_validate_derivative(pmc, inst, solution, deriv, args)
    
if not args.no_export:
    deriv = export_json(args, pmc, T, inst, solution, deriv)

current_time = datetime.now().strftime("%H:%M:%S")
print('\npMC code ended at {}\n'.format(current_time))
print('=============================================')