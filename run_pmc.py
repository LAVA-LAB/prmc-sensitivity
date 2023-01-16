# %run "~/documents/sensitivity-prmdps/prmdp-sensitivity-git/run_pmc.py"

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

from core.commons import tocDiff

# Load PRISM model with STORM
root_dir = os.path.dirname(os.path.abspath(__file__))

# Parse arguments
args = parse_inputs()

SWITCH = 3

if SWITCH == 1:
    args.model = 'models/pdtmc/parametric_die.pm'
    args.formula = 'P=? [F s=7 & d=2]'

elif SWITCH == 2:
    args.model = 'models/POMDP/maze/maze_simple_extended_m5.drn'
    args.formula = 'R=? [F "goal"]'

elif SWITCH == 3:
    args.model = 'models/PMC/brp/brp_512_5.pm'
    args.formula = 'P=? [ F s=5 ]'
    args.parameters ='models/PMC/brp/brp.json'
    
elif SWITCH == 4:
    args.model = 'models/POMDP/drone/pomdp_drone_4-2-mem1-simple.drn'
    args.formula = 'P=? ["notbad" U "goal"]'
    
elif SWITCH == 5:
    args.model = 'models/satellite/pomdp_prob_36_sat_065_dist_1_obs_diff_orb_len_40.drn'
    args.formula = 'P=? [F "goal"]'

current_time = datetime.now().strftime("%H:%M:%S")
print('Program started at {}'.format(current_time))
print('\n',tabulate(vars(args).items(), headers=["Argument", "Value"]),'\n')

model, terminal_states, instantiated_model, parameters, point, result, params2states = parse_pmdp(
    path = str(Path(root_dir, args.model)),
    formula = args.formula,
    args = args,
    param_path = str(Path(root_dir, args.parameters)))
print(model)

import scipy.sparse as sparse

row = []
col = []
val = []

print('Start defining J...')

subpoint = {}

for state in model.states:
        for action in state.actions:
            for transition in action.transitions:
                
                ID = (state.id, action.id, transition.column)
                
                # Gather variables related to this transition
                var = transition.value().gather_variables()
                
                # Get valuation for only those parameters
                subpoint[ID] = {v: point[v] for v in var}
                
                if state.id not in terminal_states:
                
                    value = float(transition.value().evaluate(subpoint[ID]))
                    if value != 0:
                
                        # Add to sparse matrix
                        row += [state.id]
                        col += [transition.column]
                        val += [value]
      
J = sparse.identity(len(model.states)) - sparse.csc_matrix((val, (row, col)), shape=(len(model.states), len(model.states)))  
      
print('Start defining Ju...')

tocDiff()

Ju = {}



row = []
col = []
val = []

sol = result.get_values()

for q,p in enumerate(parameters):
    
    if q % 100 == 0:
        print('#{}, parameter: {}'.format(q,p))
    
    for state in params2states[p]:
    # for state in model.states:
        
        # If the value in this state is zero, we can skip it anyways
        if sol[state.id] != 0:        
            for action in state.actions:
                cur_val = 0
                
                for transition in action.transitions:
                    
                    ID = (state.id, action.id, transition.column)
                    
                    value = float(transition.value().derive(p).evaluate(subpoint[ID]))
                    
                    cur_val += value * sol[transition.column]
                    
                if cur_val != 0:
                    row += [state.id]
                    col += [q]
                    val += [cur_val]
                        
                
                
    # Ju[q] = sparse.csc_matrix((val, (row, col)), shape=(len(model.states), len(model.states))) @ sparse_result
   
tocDiff()
             
# JU = -np.column_stack([g for g in Ju.values()])
JU = -sparse.csc_matrix((val, (row, col)), shape=(len(model.states), len(parameters)))




# for q,p in enumerate(parameters):
    
#     if q % 100 == 0:
#         print('#{}, parameter: {}'.format(q,p))
    
#     row = []
#     col = []
#     val = []
    
#     for state in params2states[p]:
#     # for state in model.states:
        
#         for action in state.actions:
#             for transition in action.transitions:
                
#                 ID = (state.id, action.id, transition.column)
                
#                 value = float(transition.value().derive(p).evaluate(subpoint[ID]))
                
#                 if value != 0:
#                     row += [state.id]
#                     col += [transition.column]
#                     val += [value]
                
#     Ju[q] = sparse.csc_matrix((val, (row, col)), shape=(len(model.states), len(model.states))) @ result.get_values()
   
# tocDiff()
             
# JU = -np.column_stack([g for g in Ju.values()])







from core.sensitivity import solve_cvx, solve_cvx_gurobi

print('Start solving...')

sI = {'s': np.array([0]), 
      'p': np.array([1])}

# K, v, time = solve_cvx(J, JU, sI, 1, solver='GUROBI', verbose=True)
# print('Parameter ID: {}; Gradient: {}; Solver time: {}'.format(K,v,time))
# print('Chosen parameter is:', parameters[K])

K, v, time = solve_cvx_gurobi(J, JU, sI, 1)
print('Parameter ID: {}; Gradient: {}; Solver time: {}'.format(K,v,time))
print('Chosen parameter is:', parameters[K])

assert False










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