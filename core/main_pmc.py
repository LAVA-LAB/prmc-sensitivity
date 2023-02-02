from core.classes import PMC
from core.sensitivity import solve_cvx_gurobi
from core.baseline_gradient import explicit_gradient
from core.export import export_json

import numpy as np
import scipy.sparse as sparse
import time
from gurobipy import GRB
import random
import stormpy
import sys

def run_pmc(args, model_path, param_path, verbose):
    
    T = {}
    
    start_time = time.time()
    
    pmc = PMC(model_path = model_path, args = args, )
    
    inst = {}
    
    inst['valuation'], inst['sample_size'] = pmc.load_instantiation(args = args, param_path = param_path)

    T['parse_model'] = time.time() - start_time
    print('\n',pmc.model)

    ### Verify model

    print('Model checking pMC...')

    start_time = time.time()
    instantiated_model, inst['point'] = pmc.instantiate(inst['valuation'])
    params2states = pmc.get_parameters_to_states()
    T['instantiate'] = time.time() - start_time

    start_time = time.time()
    J, subpoint  = define_sparse_LHS(pmc.model, inst['point'])
    T['build_matrices'] = time.time() - start_time
    
    print('DET:', np.linalg.det(J.toarray()))
    
    if args.goal_label is not None:
        pmc.reward = get_pmdp_reward_from_label(pmc.model, args.goal_label)
    else:
        pmc.reward = get_pmdp_reward_vector(pmc.model, inst['point'])

    # If we want to skip pMC derivative computation, return here
    if args.no_pMC:
        return pmc, T, inst, None, None

    # Solve pMC (either use Storm, or simply solve equation system)
    start_time = time.time()
    if args.pMC_engine == 'storm':
        print('- Verify with Storm...')
        result = verify_pmdp_storm(instantiated_model, pmc.properties)
    else:
        print('- Verify by solving sparse equation system...')
        result = sparse.linalg.spsolve(J, pmc.reward)  
    T['verify'] = time.time() - start_time
        
    print('Range of solutions: [{}, {}]'.format(np.min(result), np.max(result)))
    print('Solution in initial state: {}\n'.format(result[pmc.sI['s']] @ pmc.sI['p']))

    start_time = time.time()
    Ju = define_sparse_RHS(pmc.model, pmc.parameters, params2states, result, subpoint)
    T['build_matrices'] += time.time() - start_time

    # Upper bound number of derivatives to the number of parameters
    args.num_deriv = min(args.num_deriv, len(pmc.parameters))

    ### Compute K most important parameters

    deriv = {}

    print('Compute parameter importance via LP (GurobiPy)...')
    start_time = time.time()

    deriv['LP_idxs'], deriv['LP'] = solve_cvx_gurobi(J, Ju, pmc.sI, args.num_deriv,
                                        direction=GRB.MAXIMIZE, verbose=verbose)

    deriv['LP_pars'] = pmc.parameters[ deriv['LP_idxs']][0].name

    T['solve_LP'] = time.time() - start_time   
    print('- LP solved in: {:.3f} sec.'.format(T['solve_LP']))
    print('- Obtained derivatives are {} for parameters {}'.format(deriv['LP'],  deriv['LP_pars']))

    if args.explicit_baseline:
        print('-- Execute baseline: compute all gradients explicitly')
        
        deriv['explicit'], T['solve_explicit_one'], T['solve_explicit_all'] = \
            explicit_gradient(pmc, args, J, Ju)
            
    # Empirical validation of gradients
    solution = result[pmc.sI['s']] @ pmc.sI['p']
    
    if not args.no_gradient_validation:

        print('\nValidation by perturbing parameters by +{}'.format(args.validate_delta))
        
        deriv['validate'] = np.zeros(args.num_deriv, dtype=float)
        deriv['RelDiff']  = np.zeros(args.num_deriv, dtype=float)
        
        for q,x in enumerate(pmc.parameters[deriv['LP_idxs']]):
            
            # Increment this parameter by the given delta
            inst['valuation'][x.name] += args.validate_delta
            
            # instantiate model
            instantiated_model, point = pmc.instantiate(inst['valuation'])
            
            # Verify pMC (either use Storm, or simply solve equation system)
            if args.pMC_engine == 'storm':
                # print('- Verify with Storm...')
                result = verify_pmdp_storm(instantiated_model, pmc.properties)
                
            else:
                # print('- Verify by solving sparse equation system...')
                # R = get_pmdp_reward_vector(pmc.model, point)
                J_delta, subpoint  = define_sparse_LHS(pmc.model, point)
                result = sparse.linalg.spsolve(J_delta, pmc.reward)
                
            # Extract solution
            solution_new = result[pmc.sI['s']] @ pmc.sI['p']
            
            # Compute derivative
            deriv['validate'][q] = (solution_new-solution) / args.validate_delta
            
            # Determine difference in %
            if deriv['LP'][q] != 0:
                deriv['RelDiff'][q] = (deriv['validate'][q]-deriv['LP'][q])/deriv['LP'][q]
            
            print('- Parameter {}, LP: {:.3f}, val: {:.3f}, diff: {:.3f}'.format(x,  deriv['validate'][q], deriv['LP'][q], deriv['RelDiff'][q]))
            
            inst['valuation'][x.name] -= args.validate_delta
            
    if not args.no_export:
        export_json(args, pmc, T, inst, solution, deriv)
            
    return pmc, T, inst, solution, deriv



def get_pmdp_reward_from_label(model, label):
    
    R = np.zeros(len(model.states))
    
    all_labels = set({})
    
    for s in model.states:
        all_labels.update(model.labels_state(s.id))
        if label in model.labels_state(s.id):
            R[s.id] = 1
            
    # print(all_labels)
    # assert False
    
    return R



def get_pmdp_reward_vector(model, point):
    
    reward_model = next(iter(model.reward_models.values()))
    R = np.zeros(len(model.states))
    
    for s in model.states:
        if not model.is_sink_state(s):
            if reward_model.has_state_rewards:
                R[s.id] = float(reward_model.get_state_reward(s.id).evaluate(point))
            elif reward_model.has_state_action_rewards:
                R[s.id] = float(reward_model.get_state_action_reward(s.id).evaluate(point))
            else:
                sys.exit()
                
    return R



def verify_pmdp_storm(instantiated_model, properties):
    
    # Compute model checking result
    result = stormpy.model_checking(instantiated_model, properties[0])
    array  = np.array(result.get_values(), dtype=float)

    return array



def define_sparse_LHS(model, point):

    subpoint = {}    
    
    row = []
    col = []
    val = []
    
    for state in model.states:
        for action in state.actions:
            for transition in action.transitions:
                
                ID = (state.id, action.id, transition.column)
                
                # Gather variables related to this transition
                var = transition.value().gather_variables()
                
                # Get valuation for only those parameters
                subpoint[ID] = {v: point[v] for v in var}
                
                if not model.is_sink_state(state.id):
                
                    value = float(transition.value().evaluate(subpoint[ID]))
                    if value != 0:
                
                        # Add to sparse matrix
                        row += [state.id]
                        col += [transition.column]
                        val += [value]
        
    # Create sparse matrix for left-hand side
    J = sparse.identity(len(model.states)) - sparse.csc_matrix((val, (row, col)), shape=(len(model.states), len(model.states)))
    
    return J, subpoint
    


def define_sparse_RHS(model, parameters, params2states, sols, subpoint):
    
    row = []
    col = []
    val = []
    
    for q,p in enumerate(parameters):
        
        for state in params2states[p]:
            
            # If the value in this state is zero, we can skip it anyways
            if sols[state.id] != 0:        
                for action in state.actions:
                    cur_val = 0
                    
                    for transition in action.transitions:
                        
                        ID = (state.id, action.id, transition.column)
                        
                        value = float(transition.value().derive(p).evaluate(subpoint[ID]))
                        
                        cur_val += value * sols[transition.column]
                        
                    if cur_val != 0:
                        row += [state.id]
                        col += [q]
                        val += [cur_val]
                 
    # Create sparse matrix for right-hand side
    Ju = -sparse.csc_matrix((val, (row, col)), shape=(len(model.states), len(parameters)))
    
    return Ju