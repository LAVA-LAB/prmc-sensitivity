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
# from scikits.umfpack import spsolve as spsolve_umf
import cvxpy as cp
import os
import time
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime
from gurobipy import GRB
import random

# Load PRISM model with STORM
root_dir = os.path.dirname(os.path.abspath(__file__))

# Parse arguments
args = parse_inputs()

# args.model = 'models/slipgrid/pmc_size=100_params=1000_seed=0_conv.drn'
# args.parameters = 'models/slipgrid/pmc_size=100_params=1000_seed=0.json'
# args.formula = 'Rmin=? [F \"goal\"]'
# args.num_deriv = 1

# args.instance = 'Dummy_drn'

### Parse model

current_time = datetime.now().strftime("%H:%M:%S")
print('Program started at {}'.format(current_time))
print('\n',tabulate(vars(args).items(), headers=["Argument", "Value"]),'\n')

start_time = time.time()

model, properties, parameters, valuation, sample_size = parse_pmdp(
    path = Path(root_dir, args.model),
    args = args,
    param_path = Path(root_dir, args.parameters) if args.parameters else False)

instantiated_model, point, params2states = instantiate_pmdp(model, properties, parameters, valuation)

time_parse_model = time.time() - start_time
print('\n',model)

# Define initial state
sI = {'s': np.array(model.initial_states), 
      'p': np.full(len(model.initial_states), 1/len(model.initial_states))}

### Verify model

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

# Upper bound number of derivatives to the number of parameters
args.num_deriv = min(args.num_deriv, len(parameters))

### Compute K most important parameters

print('\n--------------------------------------------------------------')
print('Solve LP using Gurobi...')
start_time = time.time()
K, v, solve_time = solve_cvx_gurobi(J, Ju, sI, args.num_deriv, direction=GRB.MAXIMIZE, verbose=False)
print('- LP solved in: {}'.format(solve_time))
# If the number of desired parameters >1, then we still need to obtain their values
if args.num_deriv > 1:
    # Deriv = np.zeros_like(K, dtype=float)
    
    # for i,k in enumerate(K):
    #     Deriv[i] = sparse.linalg.spsolve(J, -Ju[:,k])[sI['s']].T @ sI['p']
    #     print('-- Derivative for parameter {} is: {:.3f}'.format(parameters[k], Deriv[i]))
        
    Deriv = sparse.linalg.spsolve(J, -Ju[:,K])[sI['s']].T @ sI['p']
        
else:
    Deriv = np.array([v])
    print('-- Derivative for parameter {} is: {:.3f}'.format(parameters[K], v))
time_solve_LP = time.time() - start_time   
print('--------------------------------------------------------------\n')    

### Baseline of computing parameters explicitly
if args.explicit_baseline:
    
    start_time = time.time()
    
    # Select N random parameters
    idxs = np.arange(len(parameters))
    random.shuffle(idxs)
    sample_idxs = idxs[:min(len(parameters), 100)]
    
    Deriv_expl = np.zeros(len(sample_idxs), dtype=float)
    
    for i,(q,x) in enumerate(zip(sample_idxs, parameters[sample_idxs])):
        
        Deriv_expl[i] = sparse.linalg.spsolve(J, -Ju[:,q])[sI['s']] @ sI['p']
        
    time_solve_explicit = (time.time() - start_time) * (len(parameters) / len(sample_idxs))

# Empirical validation of gradients
sol = sols[sI['s']] @ sI['p']




print('\nValidate derivatives with delta of {}'.format(args.validate_delta))

gradient_validate = {}
for q,x in enumerate(parameters[K]):
    
    valuation[x.name] += args.validate_delta
    
    instantiated_model, point, params2states = instantiate_pmdp(model, properties, parameters, valuation)
    
    # Solve pMC (either use Storm, or simply solve equation system)
    if args.pMC_engine == 'storm':
        # print('- Verify with Storm...')
        result = verify_pmdp(instantiated_model, properties)
        sols_new = np.array(result.get_values(), dtype=float)
        
    else:
        # print('- Verify by solving sparse equation system...')
        R = get_pmdp_reward_vector(model, point)
        J_delta, subpoint  = define_sparse_LHS(model, point)
        sols_new = sparse.linalg.spsolve(J_delta, R)
        
    sol_new = sols_new[sI['s']] @ sI['p']
    
    grad = (sol_new-sol) / args.validate_delta
    gradient_validate[x.name] = grad
    
    print('-- Parameter {}, LP: {:.3f}, val: {:.3f}, diff: {:.3f}'.format(x, grad, Deriv[q], (grad-Deriv[q])/Deriv[q]))
    
    valuation[x.name] -= args.validate_delta

PD_gradient = pd.Series(gradient_validate)

inf = PD_gradient.min()
inf_par = PD_gradient.idxmin()
sup = PD_gradient.max()
sup_par = PD_gradient.idxmax()

print('\nMinimum gradient: {} for parameter {}'.format(inf, inf_par))
print('Maximum gradient: {} for parameter {}'.format(sup, sup_par))




# Export results
if not args.no_export:
    print('\n- Export results...')

    import json
    
    OUT = {
           'instance': args.model if not args.instance else args.instance,
           'formula': args.formula,
           'engine': args.pMC_engine,
           'states': model.nr_states,
           'transitions': model.nr_transitions,
           'parameters': len(valuation),
           'solution': sol,
           'parse [s]': np.round(time_parse_model, 5),
           'verify [s]': np.round(time_verify, 5),
           'deriv. build LP [s]': np.round(time_build_LP, 5),
           'deriv. solve LP [s]': np.round(time_solve_LP, 5),
           'LP max. partial': list(np.round(Deriv, 5)),
           'LP max. param': [p.name for p in parameters[K]],
           'Val. max partial': sup
           }
    
    if args.explicit_baseline:
        OUT['deriv. explicit [s]'] = time_solve_explicit
    
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