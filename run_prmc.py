from core.classes import PMC

from core.pmc_functions import pmc_load_instantiation, pmc_instantiate, assert_probabilities
from core.verify_pmc import pmc_verify, pmc_get_reward
from core.export import export_json, timer
from core.prmc_functions import pmc2prmc, prmc_verify, prmc_derivative_LP, prmc_validate_derivative
from core.parser import parse_main

from pathlib import Path
from tabulate import tabulate
from datetime import datetime

import os
import numpy as np

# Parse arguments
args = parse_main()

# args.model = 'models/pdtmc/brp16_2.pm'
# args.formula = 'P=? [ F s=5 ]'
# args.default_valuation = 0.9
# args.goal_label = {'(s = 5)'}

# args.model = 'models/pmdp/wlan/wlan0_param.nm'
# args.formula = 'R{"time"}max=? [ F s1=12 | s2=12 ]'
# args.default_valuation = 0.01
# args.robust_bound = 'upper'

# args.model = 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn'
# args.formula = 'P=? ["notbad" U "goal"]'
# args.goal_label = {'goal','notbad'}
# args.robust_bound = 'upper'

# args.model = 'models/pmdp/CSMA/csma2_4_param.nm'
# args.formula = 'R{"time"}max=? [ F "all_delivered" ]'
# args.default_valuation = '0.05'
# args.robust_bound = 'upper'

# args.model = 'models/pdtmc/nand5_10.pm'
# args.formula = 'P=? [F \"target\" ]'
# args.parameters = 'models/pdtmc/nand.json'
# args.goal_label = {'target'}
# args.robust_bound = 'upper'

# args.model = 'models/pmdp/coin/coin4.pm'
# args.formula = 'Pmin=? [ F "all_coins_equal_1" ]'
# args.default_valuation = 0.4
# args.goal_label = {'all_coins_equal_1'}
# args.robust_bound = 'upper'

# args.model = 'models/sttt-drone/drone_model.nm'
# args.formula = 'Pmax=? [F attarget ]'
# args.default_valuation = 0.07692307692
# args.goal_label = {'(((x > (15 - 2)) & (y > (15 - 2))) & (z > (15 - 2)))'}
# args.robust_bound = 'upper'

# Load PRISM model with STORM
args.root_dir = os.path.dirname(os.path.abspath(__file__))

args.beta_penalty = 0
args.validate_delta = 1e-1

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
args.robust_dependencies = 'parameter' # Can be 'none' or 'parameter'
args.default_sample_size = 1000

print('Convert pMC to prMC...')
prmc = pmc2prmc(pmc.model, pmc.parameters, pmc.scheduler_prob, inst['point'], inst['sample_size'], args, args.verbose, T)

print('Verify prMC...')
P, solution = prmc_verify(prmc, pmc, args, args.verbose, T)

print('\nSet up sensitivity computation...')
G, deriv = prmc_derivative_LP(prmc, pmc, P, args, T)

if not args.no_gradient_validation:
    deriv = prmc_validate_derivative(prmc, pmc, inst, solution, deriv, 
                     args.validate_delta, args.robust_bound, args.beta_penalty)

if not args.no_export:
    export_json(args, prmc, T, inst, solution, deriv)

current_time = datetime.now().strftime("%H:%M:%S")
print('\nprMC code ended at {}\n'.format(current_time))
print('=============================================')