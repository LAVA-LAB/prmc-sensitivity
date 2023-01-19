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
from gurobipy import GRB

from core.commons import tocDiff

# Load PRISM model with STORM
root_dir = os.path.dirname(os.path.abspath(__file__))

# Parse arguments
args = parse_inputs()

SWITCH = 0

if SWITCH == 0:
    args.model = 'models/PMC/slipgrid_fixed/slipgrid_fixed.nm'
    args.formula = 'R=? [F "goal"]'
    args.parameters = 'models/PMC/slipgrid_fixed/slipgrid_fixed_mle.json'

elif SWITCH == 1:
    args.model = 'models/pdtmc/parametric_die.pm'
    args.formula = 'P=? [F s=7 & d=2]'
    args.parameters = 'models/pdtmc/parametric_die.json'

elif SWITCH == 2:
    args.model = 'models/POMDP/maze/maze_simple_extended_m5.drn'
    args.formula = 'R=? [F "goal"]'

elif SWITCH == 3:
    args.model = 'models/PMC/brp/brp_512_5.pm'
    args.formula = 'P=? [ F s=5 ]'
    args.parameters = 'models/PMC/brp/brp.json'
    
elif SWITCH == 4:
    args.model = 'models/POMDP/drone/pomdp_drone_4-2-mem1-simple.drn'
    args.formula = 'P=? ["notbad" U "goal"]'
    
elif SWITCH == 5:
    args.model = 'models/satellite/pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06_sparse_full.drn'
    args.formula = 'P=? [F "goal"]'
    
elif SWITCH == 6:
    args.model = 'models/satellite/pomdp_prob_36_sat_065_dist_1_obs_diff_orb_len_40.drn'
    args.formula = 'P=? [F "goal"]'
    args.default_valuation = 0.2

current_time = datetime.now().strftime("%H:%M:%S")
print('Program started at {}'.format(current_time))
print('\n',tabulate(vars(args).items(), headers=["Argument", "Value"]),'\n')

model, properties, terminal_states, instantiated_model, parameters, valuation, point, result, params2states = parse_pmdp(
    path = str(Path(root_dir, args.model)),
    formula = args.formula,
    args = args,
    param_path = str(Path(root_dir, args.parameters)) if args.parameters else False)
print(model)

# Retrieve model checking solution
sol = np.array(result.get_values())

# Define initial state
sI = {'s': np.array(model.initial_states), 
      'p': np.full(len(model.initial_states), 1/len(model.initial_states))}

print('Range of solutions: [{}, {}]'.format(np.min(sol), np.max(sol)))
print('Solution in initial state: {}\n'.format(sol[sI['s']] @ sI['p']))

# Get only terminal states, and save mapping from this list to absolute state indices
# nonterm_states      = [True if i not in terminal_states else False for i in np.arange(model.nr_states)]
# new_old_state_map   = {j: i for j,i in enumerate([i for i in np.arange(model.nr_states) if i not in terminal_states])}
# old_new_state_map   = inv_map = {v: k for k, v in new_old_state_map.items()}



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

row = []
col = []
val = []

for q,p in enumerate(parameters):
    
    if q % 100 == 0:
        print('#{}, parameter: {}'.format(q,p))
    
    for state in params2states[p]:
        
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
                
tocDiff()
             
# JU = -np.column_stack([g for g in Ju.values()])
Ju = -sparse.csc_matrix((val, (row, col)), shape=(len(model.states), len(parameters)))

from core.sensitivity import solve_cvx, solve_cvx_gurobi

print('Start solving...')

# K, v, time = solve_cvx(J, Ju, sI, 1, solver='GUROBI', direction=cp.maximize, verbose=True)
# print('Parameter ID: {}; Gradient: {}; Solver time: {}'.format(K,v,time))
# print('Chosen parameter is:', parameters[K])

K, v, time = solve_cvx_gurobi(J, Ju, sI, 1, direction=GRB.MAXIMIZE, verbose=True)
print('Parameter ID: {}; Gradient: {}; Solver time: {}'.format(K,v,time))
print('Chosen parameter is:', parameters[K])


#####

# %%
# Empirical validation of gradients

from models.pmdp import instantiate_pmdp
import stormpy
import pandas as pd

delta = 1e-8
sol_old = sol[sI['s']] @ sI['p']

gradient_validate = {}
for q,x in enumerate(parameters[K]):
    
    valuation[x.name] += delta
    
    # Instantiate parameters
    instantiated_model, params2states, point = instantiate_pmdp(parameters, valuation, model)
    
    # Compute model checking result
    result_delta = stormpy.model_checking(instantiated_model, properties[0])
    result_array = np.array(result_delta.get_values())
    sol_new = result_array[sI['s']] @ sI['p']
    grad = (sol_new-sol_old) / delta
    gradient_validate[x.name] = grad
    
    if q % 100 == 0:
        print('Validated #{}, parameter: {}, gradient: {}'.format(q,x,grad))
    
    valuation[x.name] -= delta

PD_gradient = pd.Series(gradient_validate)

inf = PD_gradient.min()
inf_par = PD_gradient.idxmin()
sup = PD_gradient.max()
sup_par = PD_gradient.idxmax()

print('Minimum gradient: {} for parameter {}'.format(inf, inf_par))
print('Maximum gradient: {} for parameter {}'.format(sup, sup_par))

assert False

# %%








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