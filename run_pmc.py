# %run "~/documents/sensitivity-prmdps/prmdp-sensitivity-git/run_pmc.py"

from core.cvx import cvx_verification
from core.parse_inputs import parse_inputs
from models.prmdp import parse_prmdp
from models.pmdp import get_pmdp_reward_vector, parse_pmdp, instantiate_pmdp, verify_pmdp

from core.sensitivity import gradient, solve_cvx, solve_cvx_gurobi
from core.pMC_LP import define_sparse_LHS, define_sparse_RHS

import pandas as pd
import numpy as np
import scipy.sparse as sparse
import cvxpy as cp
import os
import time
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime
from gurobipy import GRB

from core.commons import tocDiff

# Load PRISM model with STORM
root_dir = os.path.dirname(os.path.abspath(__file__))

# Parse arguments
args = parse_inputs()

#args.model = 'models/slipgrid/pmc_size=160_params=500_seed=0.drn'
#args.formula = 'R=? [F "goal"]'
#args.parameters = 'models/slipgrid/pmc_size=80_params=500_seed=0.json'
#args.terminal_label = 'goal'
#args.validate_delta = 0.001

# SWITCH = 0

# if SWITCH == 0:
#     # args.model = 'models/PMC/slipgrid_fixed/slipgrid_fixed.nm'
#     args.model   = 'models/slipgrid/input/pmc_size=80_params=5000_seed=0.nm'
#     args.formula = 'R=? [F "goal"]'
#     args.parameters = 'models/slipgrid/input/pmc_size=80_params=5000_seed=0.json'

# elif SWITCH == 1:
#     args.model = 'models/pdtmc/parametric_die.pm'
#     args.formula = 'P=? [F s=7 & d=2]'
#     args.parameters = 'models/pdtmc/parametric_die.json'

# elif SWITCH == 2:
#     args.model = 'models/POMDP/maze/maze_simple_extended_m5.drn'
#     args.formula = 'R=? [F "goal"]'

# elif SWITCH == 3:
#     args.model = 'models/PMC/brp/brp_512_5.pm'
#     args.formula = 'P=? [ F s=5 ]'
#     args.parameters = 'models/PMC/brp/brp.json'
    
# elif SWITCH == 4:
#     args.model = 'models/POMDP/drone/pomdp_drone_4-2-mem1-simple.drn'
#     args.formula = 'P=? ["notbad" U "goal"]'
    
# elif SWITCH == 5:
#     args.model = 'models/satellite/pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06_sparse_full.drn'
#     args.formula = 'P=? [F "goal"]'
    
# elif SWITCH == 6:
#     args.model = 'models/satellite/pomdp_prob_36_sat_065_dist_1_obs_diff_orb_len_40.drn'
#     args.formula = 'P=? [F "goal"]'
#     args.default_valuation = 0.2

current_time = datetime.now().strftime("%H:%M:%S")
print('Program started at {}'.format(current_time))
print('\n',tabulate(vars(args).items(), headers=["Argument", "Value"]),'\n')

start_time = time.time()

model, properties, parameters, valuation, point, instantiated_model, params2states = parse_pmdp(
    path = str(Path(root_dir, args.model)),
    formula = args.formula,
    args = args,
    param_path = str(Path(root_dir, args.parameters)) if args.parameters else False)

time_parse_model = time.time() - start_time
print('\n',model)

# Define initial state
sI = {'s': np.array(model.initial_states), 
      'p': np.full(len(model.initial_states), 1/len(model.initial_states))}

print('Start defining J...')
start_time = time.time()
J, subpoint  = define_sparse_LHS(model, point)
time_build_LP = time.time() - start_time

# Solve pMC (either use Storm, or simply solve equation system)
start_time = time.time()
if args.pMC_engine == 'storm':
    print('- Verify with Storm...')
    result = verify_pmdp(instantiated_model, properties)
    sols = np.array(result.get_values(), dtype=float)
else:
    print('- Verify by solving sparse equation system...')
    R = get_pmdp_reward_vector(model, point)
    sols = sparse.linalg.spsolve(J, R)  
time_verify = time.time() - start_time
    
print('Range of solutions: [{}, {}]'.format(np.min(sols), np.max(sols)))
print('Solution in initial state: {}\n'.format(sols[sI['s']] @ sI['p']))

print('Start defining Ju...')
start_time = time.time()
Ju = define_sparse_RHS(model, parameters, params2states, sols, subpoint)
time_build_LP += time.time() - start_time

print('Start solving...')

# K, v, time = solve_cvx(J, Ju, sI, 1, solver='GUROBI', direction=cp.Maximize, verbose=True)
# print('Parameter ID: {}; Gradient: {}; Solver time: {}'.format(K,v,time))
# print('Chosen parameter is:', parameters[K])

start_time = time.time()
K, v, solve_time = solve_cvx_gurobi(J, Ju, sI, 1, direction=GRB.MAXIMIZE, verbose=True)
time_solve_LP = time.time() - start_time
print('Parameter ID: {}; Gradient: {}; Solver time: {}'.format(K,v,time))
print('Chosen parameter is:', parameters[K])

# %%
# Empirical validation of gradients

sol_old = sols[sI['s']] @ sI['p']

print('- Validate derivatives with delta of {}'.format(args.validate_delta))

gradient_validate = {}
for q,x in enumerate(parameters[K]):
    
    valuation[x.name] += args.validate_delta
    
    instantiated_model, point, params2states = instantiate_pmdp(model, properties, parameters, valuation)
    
    # Solve pMC (either use Storm, or simply solve equation system)
    if args.pMC_engine == 'storm':
        print('- Verify with Storm...')
        result = verify_pmdp(instantiated_model, properties)
        sols_new = np.array(result.get_values(), dtype=float)
        
    else:
        print('- Verify by solving sparse equation system...')
        R = get_pmdp_reward_vector(model, point)
        J_delta, subpoint  = define_sparse_LHS(model, point)
        sols_new = sparse.linalg.spsolve(J_delta, R)
        
    sol_new = sols_new[sI['s']] @ sI['p']
    
    grad = (sol_new-sol_old) / args.validate_delta
    gradient_validate[x.name] = grad
    
    if q % 100 == 0:
        print('Validated #{}, parameter: {}, gradient: {}'.format(q,x,grad))
    
    valuation[x.name] -= args.validate_delta

PD_gradient = pd.Series(gradient_validate)

inf = PD_gradient.min()
inf_par = PD_gradient.idxmin()
sup = PD_gradient.max()
sup_par = PD_gradient.idxmax()

print('Minimum gradient: {} for parameter {}'.format(inf, inf_par))
print('Maximum gradient: {} for parameter {}'.format(sup, sup_par))

# %%

# Export results
if not args.no_export:
    print('\n- Export results...')

    import json
    
    OUT = {
           'instance': args.model,
           'formula': args.formula,
           'engine': args.pMC_engine,
           'states': model.nr_states,
           'transitions': model.nr_transitions,
           'parameters': len(valuation),
           'parse [s]': time_parse_model,
           'verify [s]': time_verify,
           'build LP [s]': time_build_LP,
           'solve LP [s]': time_solve_LP,
           'LP max. partial': v,
           'LP max. param': [p.name for p in parameters[K]],
           'Val. max partial': sup
           }
    
    output_path = Path(root_dir, args.output_folder)
    if not os.path.exists(output_path):
    
       # Create a new directory because it does not exist
       os.makedirs(output_path)
    
    dt = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    out_file = os.path.join(args.output_folder, Path(args.model).stem + dt + '.json')
    with open(str(Path(root_dir, out_file)), "w") as outfile:
        json.dump(OUT, outfile)
        
current_time = datetime.now().strftime("%H:%M:%S")
print('\nProgram ended at {}\n'.format(current_time))