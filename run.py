# %run "~/documents/sensitivity-prmdps/prmdp-sensitivity-git/run.py"

from core.cvx import cvx_verification
from core.parse_inputs import parse_inputs
from models.prmdp import parse_prmdp
from models.pmdp import parse_pmdp

from core.sensitivity import gradient

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime

# Load PRISM model with STORM
root_dir = os.path.dirname(os.path.abspath(__file__))

# Parse arguments
args = parse_inputs()

args.model = 'models/mdp/slipgrid.nm'
args.formula = 'Rmin=? [F "goal"]'

# args.model = 'models/dtmc/dummy.nm'
# args.formula = None

# args.validate_gradients = True

current_time = datetime.now().strftime("%H:%M:%S")
print('Program started at {}'.format(current_time))
print('\n',tabulate(vars(args).items(), headers=["Argument", "Value"]),'\n')

M = parse_prmdp(path = str(Path(root_dir, args.model)),
               formula = args.formula,
               policy = 'random',
               args = args)
print(M)

# Create object for computing gradients
G = gradient(M)


# Verify loaded model by solving the optimization program
CVX = cvx_verification(M, verbose=True, pars_as_expressions=False)
CVX.solve(solver = args.solver, store_initial = True, verbose=True)
print('Optimal solution:', CVX.prob.value)

# Check if complementary slackness is satisfied
CVX.check_complementary_slackness()

# Update gradient object with current solution
G.update(M, CVX, mode='remove_dual')

gradients1, time = G.solve_eqsys()

M.pars_at_max = np.array([True if v.value >= M.parameters_max_value[(s,a)] else False for (s,a),v in M.parameters.items()])

k=10
print('Determine {} most important parameters...'.format(k))
K, y, v, time = G.solve_cvx(M, k, solver='GUROBI')
print('Parameter importance LP solved in {} seconds'.format(time))

# gradients2, time3 = G.solve_inversion(CVX)
# print('Via matrix inversion took', time3)

# print(time1,time2,time3)

assert False

# %%




done = False

THRESHOLD = CVX.prob.value * 0.9
MODE = 'maximize'
STEP = 0.1
z = 0

PARS_CHANGED = []
PARS_TO_INCR = min(1, len(M.parameters))

PARS_AT_MAX = np.array([True if v.value >= M.parameters_max_value[(s,a)] else False for (s,a),v in M.parameters.items()])

while not done:
    
    # Check if complementary slackness is satisfied
    CVX.check_complementary_slackness()
    
    ######
    
    
    
    # Increment robustness/uncertainty in k least sensitive parameters
    iSub = np.arange(len(M.parameters))[~PARS_AT_MAX]
    
    time = GRAD.solve_cvx(M, PARS_AT_MAX, solver = args.solver)
    c = np.argmax(GRAD.cvx['y'].value)
    print('Solved via CVX in {}'.format(time))
    print('Parameter that maximizes gradient: id {}, tuple {}, gradient {}'.format(
            iSub[c], M.par_idx2tuple[c], GRAD.cvx['prob'].value))
    
    time = GRAD.solve()
    print('Linear equation system for {} states and {} parameters solved in {} seconds'.format(len(M.states), len(M.parameters), time))
    
    # Compute gradient of objective (solution) wrt parameters
    gradObj = M.sI['p'] @ GRAD.gradients[M.sI['s'], :].toarray()
    
    if MODE == 'maximize':
        # get k largest elements
        h = np.argpartition(gradObj[~PARS_AT_MAX], -PARS_TO_INCR)[-PARS_TO_INCR:]
    else:
        # get k smallest elements
        h = np.argpartition(gradObj[~PARS_AT_MAX], PARS_TO_INCR)[:PARS_TO_INCR]
    
    ######
    
    if args.validate_gradients:
        print('\nValidate gradients empirically by perturbing each parameter and resolving CVX problem...')
        
        VALIDATION = np.zeros(len(M.parameters))
        
        for i, THETA in enumerate(tqdm(M.parameters.values())):
            grad_empirical = CVX.delta_solve(THETA, 1e-6, args.solver, verbose = False)
            empir = np.dot(grad_empirical[M.sI['s']], M.sI['p'])
            
            VALIDATION[i] = np.mean(np.abs(gradObj[i] - empir))
            
        print('\nCumulative average absolute difference: {}'.format(np.sum(VALIDATION)))
    
    ######
    
    # For each chosen parameter
    for iAbs in iSub[h]:
        
        param_tup = M.par_idx2tuple[iAbs]
        
        PARS_CHANGED += [M.par_idx2tuple[iAbs]]
        
        # Check if this parameter will reach its maximum (if so, declare this)
        if M.parameters[M.par_idx2tuple[iAbs]].value + STEP >= M.parameters_max_value[M.par_idx2tuple[iAbs]]:
            step = M.parameters_max_value[M.par_idx2tuple[iAbs]] - M.parameters[M.par_idx2tuple[iAbs]].value
            PARS_AT_MAX[iAbs] = True
        else:
            step = STEP
        
        # Increment parameters value
        M.parameters[M.par_idx2tuple[iAbs]].value += step
        new_value = np.round(M.parameters[M.par_idx2tuple[iAbs]].value, 5)
        print('Set par. id {}, tuple {} += {} = {} (gradient {})'.format(
                iAbs, M.par_idx2tuple[iAbs], step, new_value, gradObj[iAbs]))
    
    z += 1
    
    CVX.solve(solver = args.solver, store_initial = True, verbose = args.verbose)
    print('\nValue of the measure in initial state:', CVX.prob.value)
    
    # If threshold is not satisfied, break out of loop
    if CVX.prob.value <= THRESHOLD or z >= 10:
        done = True