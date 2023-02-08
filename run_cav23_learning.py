# %run "~/documents/sensitivity-prmdps/prmdp-sensitivity-git/run_cav23_learning.py"

from core.main_pmc import run_pmc
from core.main_prmc import run_prmc
from core.parse_inputs import parse_inputs

from core.main_prmc import pmc2prmc
from core.sensitivity import gradient, solve_cvx_gurobi
from core.verify_prmc import cvx_verification_gurobi

from core.learning.exp_visits import parameter_importance_exp_visits

import time
from gurobipy import GRB
import json

import pandas as pd
import os
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
pmc, _, inst, solution_true, deriv = run_pmc(args, model_path, true_param_path, verbose = args.verbose)
print('Optimal solution under the true parameter values: {:.3f}'.format(solution_true))
del pmc, inst, deriv



#####





current_time = datetime.now().strftime("%H:%M:%S")
print('\npMC code ended at {}\n'.format(current_time))
print('=============================================')

ITERS = 10
MAX_STEPS = 1000
SAMPLES_PER_STEP = 25

ALL_SOLUTIONS = {}

for mode in ['derivative', 'exp-visits']: #, 'samples', 'random']:
    
    # RESULTS = pd.DataFrame(columns = ['Solution', 'Parameters', 'Derivatives', 'Added_samples', 'LP_time'])

    ALL_SOLUTIONS[mode] = pd.DataFrame()
    
    for seed in range(ITERS):
    
        print('>>> Start iteration {} <<<'.format(seed))
        
        args.instance = 'seed_{}'.format(seed)
        
        np.random.seed(seed) 
    
        pmc, T, inst, solution, deriv = run_pmc(args, model_path, param_path, verbose = args.verbose)
    
        SOL_LIST = []    
    
        args.uncertainty_model = "Hoeffding"
        
        prmc = pmc2prmc(pmc.model, pmc.parameters, inst['point'], inst['sample_size'], args, verbose = args.verbose)
    
        CVX_GRB = cvx_verification_gurobi(prmc, pmc.reward, args.robust_bound, verbose = args.verbose)
    
        for i in range(MAX_STEPS):
        
            ##### PARAMETER IMPORTANCE    
            
            done = False
            trials = 0
            trials_max = 3
            solver_verbose = args.verbose
            
            
                
            # CVX_GRB.cvx.Params.NumericFocus = 3
            # CVX_GRB.cvx.Params.ScaleFlag = 1
            
            # print(CVX_GRB.cvx)
            
            CVX_GRB.solve(store_initial = True, verbose=solver_verbose)
            
            print(CVX_GRB.cvx)
            
            SOL = np.round(CVX_GRB.x_tilde[pmc.sI['s']] @ pmc.sI['p'], 2)
            SOL_LIST += [SOL]
            
            # print('Range of solutions: [{}, {}]'.format(np.min(CVX_GRB.x_tilde), np.max(CVX_GRB.x_tilde)))
            print('Solution in initial state: {}\n'.format(SOL))
            
            CVX_GRB.check_complementary_slackness(prmc)
            
            # Switch between random parameter choice or via derivative
            if mode == 'random':
    
                deriv['LP_idxs'] = np.array([np.random.randint(0, len(pmc.parameters))])
                deriv['LP'] = 0
                
                PAR = pmc.parameters[deriv['LP_idxs']]
    
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
                
                PAR = pmc.parameters[deriv['LP_idxs']]
            
            elif mode == 'expVisits':
                
                importance = parameter_importance_exp_visits(pmc, prmc, inst, CVX_GRB)
                
                PAR = [max(importance, key=importance.get)]
                
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
            
                PAR = pmc.parameters[deriv['LP_idxs']]
            
            PARAM_NAMES = []
            
            ##### OBAIN MORE SAMPLES
            for q,var in enumerate(PAR):
            
                PARAM_NAMES += [var.name]    
            
                true_prob = true_valuation[var.name]
                samples = np.random.binomial(SAMPLES_PER_STEP, true_prob)
                
                print('>> ({},{},{}) - Drawn {} more samples for parameter {} ({} positives)'.format(mode,seed, i, SAMPLES_PER_STEP, var, samples))
                
                new_sample_mean = (inst['valuation'][var.name] * inst['sample_size'][var.name] + samples) / (inst['sample_size'][var.name] + SAMPLES_PER_STEP)
                
                inst['valuation'][var.name] = new_sample_mean
                inst['sample_size'][var.name] += SAMPLES_PER_STEP
                
                # print(inst['sample_size'])
            
            ##### UPDATE PARAMETER POINT
            instantiated_model, inst['point'] = pmc.instantiate(inst['valuation'])
            
            for var in PAR:
                
                # Update mean
                prmc.update_distribution(var, inst)
                
                # Update sample size
                prmc.parameters[var].value = inst['sample_size'][var.name]
                
                print('New value:', prmc.parameters[var].value)
                
                CVX_GRB.update_parameter(prmc, var)
            
            # RESULTS.append({'Solution': SOL,
            #                 'Paramaters': PARAM_NAMES,
            #                 'Derivatives': deriv['LP'],
            #                 'Added_samples': SAMPLES_PER_STEP*len(HIGHEST_PAR),
            #                 'LP_time': T['solve_LP']}, ignore_index=True)
        
        ALL_SOLUTIONS[mode] = pd.concat([ALL_SOLUTIONS[mode], pd.Series(SOL_LIST)], axis=1)
        
    current_time = datetime.now().strftime("%H:%M:%S")
    print('\nprMC code ended at {}\n'.format(current_time))
    print('=============================================')
    
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ALL_SOLUTIONS[mode].to_csv('Learning_{}_{}.csv'.format(dt,mode), sep=';')    
        

assert False    
# %%

import matplotlib.pyplot as plt

# red dashes, blue squares and green triangles

df_merged = pd.concat([df for df in ALL_SOLUTIONS.values()], axis=1)
df_merged.columns = list(ALL_SOLUTIONS.keys())

df_merged.plot()

plt.axhline(y=solution_true, color='gray', linestyle='--')

plt.savefig('learning_curves.png')
plt.savefig('learning_curves.pdf')

plt.show()
