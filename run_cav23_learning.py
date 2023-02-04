# %run "~/documents/sensitivity-prmdps/prmdp-sensitivity-git/run_cav23_learning.py"

from core.main_pmc import run_pmc
from core.main_prmc import run_prmc
from core.parse_inputs import parse_inputs

from core.main_prmc import pmc2prmc

from core.uncertainty_models import Linf_polytope, L1_polytope, Hoeffding_interval
from core.classes import PRMC, state, action, distribution, polytope
from core.sensitivity import gradient, solve_cvx_gurobi
from core.verify_prmc import cvx_verification_gurobi
from core.baseline_gradient import explicit_gradient
from core.export import export_json

import sys
import time
from gurobipy import GRB
import json

import pandas as pd
import time
import os
import math
import numpy as np
from pathlib import Path
from tabulate import tabulate
from datetime import datetime

# Parse arguments
args = parse_inputs()

# Load PRISM model with STORM
args.root_dir = os.path.dirname(os.path.abspath(__file__))

args.model = 'models/slipgrid_learning/double_pmc_size=20_params=100_seed=0.drn'
args.parameters = 'models/slipgrid_learning/double_pmc_size=20_params=100_seed=0_mle.json'
args.formula = 'Rmin=? [F "goal"]'
args.pMC_engine = 'spsolve'
args.output_folder = 'output/learning/'
args.num_deriv = 1
args.robust_bound = 'upper'
args.no_pMC = False

### pMC execution

current_time = datetime.now().strftime("%H:%M:%S")
print('Program started at {}'.format(current_time))
print('\n',tabulate(vars(args).items(), headers=["Argument", "Value"]),'\n')

model_path = Path(args.root_dir, args.model)
param_path = Path(args.root_dir, args.parameters) if args.parameters else False

### prMC execution
true_param_file = 'models/slipgrid_learning/double_pmc_size=20_params=100_seed=0.json'
true_param_path = Path(args.root_dir, true_param_file)

# Read true parameter values
with open(str(true_param_path)) as json_file:
    true_valuation = json.load(json_file)
    
# Comptue True solution
pmc, _, inst, solution, deriv = run_pmc(args, model_path, true_param_path, verbose = args.verbose)
print('Optimal solution under the true parameter values: {:.3f}'.format(solution))
del pmc, inst, solution, deriv



#####





current_time = datetime.now().strftime("%H:%M:%S")
print('\npMC code ended at {}\n'.format(current_time))
print('=============================================')

ITERS = 5
MAX_STEPS = 500
SAMPLES_PER_STEP = 25

for mode in ['derivative', 'random', 'samples']:
    
    RESULTS = pd.DataFrame(columns = ['Solution', 'Parameters', 'Derivatives', 'Added_samples', 'LP_time'])

    ALL_SOLUTIONS = pd.DataFrame()
    
    for seed in range(ITERS):
    
        print('>>> Start iteration {} <<<'.format(seed))
        
        args.instance = 'seed_{}'.format(seed)
        
        np.random.seed(seed) 
    
        pmc, T, inst, solution, deriv = run_pmc(args, model_path, param_path, verbose = args.verbose)
    
        SOL_LIST = []    
    
        for i in range(MAX_STEPS):
        
            ##### PARAMETER IMPORTANCE    
            
            args.uncertainty_model = "Hoeffding"
            
            prmc = pmc2prmc(pmc.model, pmc.parameters, inst['point'], inst['sample_size'], args, verbose = args.verbose)
            
            done = False
            trials = 0
            trials_max = 3
            solver_verbose = args.verbose
            
            CVX_GRB = cvx_verification_gurobi(prmc, pmc.reward, args.robust_bound, verbose = args.verbose)
                
            # CVX_GRB.cvx.Params.NumericFocus = 3
            # CVX_GRB.cvx.Params.ScaleFlag = 1
            
            CVX_GRB.solve(store_initial = True, verbose=solver_verbose)
               
            SOL = np.round(CVX_GRB.x_tilde[pmc.sI['s']] @ pmc.sI['p'], 2)
            SOL_LIST += [SOL]
            
            # print('Range of solutions: [{}, {}]'.format(np.min(CVX_GRB.x_tilde), np.max(CVX_GRB.x_tilde)))
            print('Solution in initial state: {}\n'.format(SOL))
            
            CVX_GRB.check_complementary_slackness(prmc)
            
            # Switch between random parameter choice or via derivative
            if mode == 'random':
    
                deriv['LP_idxs'] = np.array([np.random.randint(0, len(pmc.parameters))])
                deriv['LP'] = 0
    
            elif mode == 'samples':
    
                minimum = 1e9            
                param   = 0
    
                for i,key in enumerate(pmc.parameters):
                    if inst['sample_size'][key.name] < minimum:
                        param = i
                        minimum = inst['sample_size'][key.name]
                
                print('- Lower parameter count is {} for {}'.format(minimum, param))
    
                deriv['LP_idxs'] = np.array([param])
                deriv['LP'] = 0
            
            else:            
    
                start_time = time.time()
                # Create object for computing gradients
                G = gradient(prmc, args.robust_bound)
                
                # Update gradient object with current solution
                G.update(prmc, CVX_GRB, mode='remove_dual')
                T['build_matrices'] = time.time() - start_time
                
                deriv = {}
                
                # print('Compute parameter importance via LP (GurobiPy)...')
                start_time = time.time()
                
                if args.robust_bound == 'lower':
                    direction = GRB.MAXIMIZE
                else:
                    direction = GRB.MINIMIZE
                    
                # print('- Shape of J matrix:', G.J.shape, G.Ju.shape)
                start_time = time.time()    
                
                deriv['LP_idxs'], deriv['LP'] = solve_cvx_gurobi(G.J, G.Ju, pmc.sI, args.num_deriv,
                                            direction=direction, verbose=args.verbose)
            
            HIGHEST_PAR = pmc.parameters[deriv['LP_idxs']]
            
            PARAM_NAMES = []
            
            ##### OBAIN MORE SAMPLES
            for q,x in enumerate(HIGHEST_PAR):
            
                PARAM_NAMES += [x.name]    
            
                true_prob = true_valuation[x.name]
                samples = np.random.binomial(SAMPLES_PER_STEP, true_prob)
                
                print('>> ({},{}) - Drawn {} more samples for parameter {} ({} positives)'.format(seed, i, SAMPLES_PER_STEP, x, samples))
                
                new_sample_mean = (inst['valuation'][x.name] * inst['sample_size'][x.name] + samples) / (inst['sample_size'][x.name] + SAMPLES_PER_STEP)
                
                inst['valuation'][x.name] = new_sample_mean
                inst['sample_size'][x.name] += SAMPLES_PER_STEP
                
                print(inst['sample_size'])
            
            ##### UPDATE PARAMETER POINT
            instantiated_model, inst['point'] = pmc.instantiate(inst['valuation'])
            
            RESULTS.append({'Solution': SOL,
                            'Paramaters': PARAM_NAMES,
                            'Derivatives': deriv['LP'],
                            'Added_samples': SAMPLES_PER_STEP*len(HIGHEST_PAR),
                            'LP_time': T['solve_LP']}, ignore_index=True)
        
        ALL_SOLUTIONS = pd.concat([ALL_SOLUTIONS, pd.Series(SOL_LIST)], axis=1)
        
    current_time = datetime.now().strftime("%H:%M:%S")
    print('\nprMC code ended at {}\n'.format(current_time))
    print('=============================================')
    
    ALL_SOLUTIONS.to_csv('Learning_{}.csv'.format(mode), sep=';')    
        
