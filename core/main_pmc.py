from models.pmdp import pMDP, get_pmdp_reward_vector, verify_pmdp_storm
from core.sensitivity import solve_cvx_gurobi
from core.pMC_LP import define_sparse_LHS, define_sparse_RHS
import numpy as np
import scipy.sparse as sparse
import os
import time
from pathlib import Path
from datetime import datetime
from gurobipy import GRB
import random

def run_pmc(args, model_path, param_path):
    
    T = {}
    
    start_time = time.time()
    
    pmc = pMDP(model_path = model_path, args = args, )
    
    inst = {}
    
    inst['valuation'], inst['sample_size'] = pmc.load_instantiation(args = args, param_path = param_path)

    T['parse_model'] = time.time() - start_time
    print('\n',pmc.model)

    ### Verify model

    print('Start defining J...')

    start_time = time.time()
    instantiated_model, inst['point'] = pmc.instantiate(inst['valuation'])
    params2states = pmc.get_parameters_to_states()
    T['instantiate'] = time.time() - start_time

    start_time = time.time()
    J, subpoint  = define_sparse_LHS(pmc.model, inst['point'])
    T['build_matrices'] = time.time() - start_time

    pmc.reward = get_pmdp_reward_vector(pmc.model, inst['point'])

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

    print('Start defining Ju...')
    start_time = time.time()
    Ju = define_sparse_RHS(pmc.model, pmc.parameters, params2states, result, subpoint)
    T['build_matrices'] += time.time() - start_time

    # Upper bound number of derivatives to the number of parameters
    args.num_deriv = min(args.num_deriv, len(pmc.parameters))

    ### Compute K most important parameters

    deriv = {}

    print('\n--------------------------------------------------------------')
    print('Solve LP using Gurobi...')
    start_time = time.time()

    deriv['LP_idxs'], deriv['LP'] = solve_cvx_gurobi(J, Ju, pmc.sI, args.num_deriv,
                                        direction=GRB.MAXIMIZE, verbose=False)

    T['solve_LP'] = time.time() - start_time   
    print('- LP solved in: {:.3f} sec.'.format(T['solve_LP']))
    print('--------------------------------------------------------------\n')

    ### Baseline of computing parameters explicitly
    if args.explicit_baseline:
        
        start_time = time.time()
        
        # Select N random parameters
        idxs = np.arange(len(pmc.parameters))
        random.shuffle(idxs)
        sample_idxs = idxs[:min(len(pmc.parameters), 100)]
        
        deriv['explicit'] = np.zeros(len(sample_idxs), dtype=float)
        
        for i,(q,x) in enumerate(zip(sample_idxs, pmc.parameters[sample_idxs])):
            
            deriv['explicit'][i] = sparse.linalg.spsolve(J, -Ju[:,q])[pmc.sI['s']] @ pmc.sI['p']
            
        T['solve_explicit_one'] = (time.time() - start_time) / len(sample_idxs)
        T['solve_explicit_all'] = T['solve_explicit_one'] * len(pmc.parameters)

    # Empirical validation of gradients
    solution = result[pmc.sI['s']] @ pmc.sI['p']
    
    if not args.no_gradient_validation:

        print('\nValidate derivatives with delta of {}'.format(args.validate_delta))
        
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
                R = get_pmdp_reward_vector(pmc.model, point)
                J_delta, subpoint  = define_sparse_LHS(pmc.model, point)
                result = sparse.linalg.spsolve(J_delta, R)
                
            # Extract solution
            solution_new = result[pmc.sI['s']] @ pmc.sI['p']
            
            # Compute derivative
            deriv['validate'][q] = (solution_new-solution) / args.validate_delta
            
            # Determine difference in %
            deriv['RelDiff'][q] = (deriv['validate'][q]-deriv['LP'][q])/deriv['LP'][q]
            
            print('-- Parameter {}, LP: {:.3f}, val: {:.3f}, diff: {:.3f}'.format(x,  deriv['validate'][q], deriv['LP'][q], deriv['RelDiff'][q]))
            
            inst['valuation'][x.name] -= args.validate_delta
            
        return pmc, T, inst, solution, deriv

def export_pmc(args, pmc, T, inst, solution, deriv):

    # Export results
    if not args.no_export:
        print('\n- Export results...')

        import json
        
        OUT = {
               'instance': args.model if not args.instance else args.instance,
               'Type': 'prMC',
               'Formula': args.formula,
               'Engine': args.pMC_engine,
               'States': pmc.model.nr_states,
               'Transitions': pmc.model.nr_transitions,
               'Parameters': len(inst['valuation']),
               #
               'Solution': np.round(solution, 6),
               'Model parse [s]': np.round(T['parse_model'], 6),
               'Model instantiate [s]': np.round(T['instantiate'], 6),
               'Model verify [s]': np.round(T['verify'], 6),
               #
               'Num. derivatives': args.num_deriv
               }
        
        if args.explicit_baseline:
            OUT['Differentiate one [s]'] = np.round(T['solve_explicit_one'], 6)
            OUT['Differentiate explicitly [s]'] = np.round(T['time_solve_explicit_all'], 3)
        
        OUT['LP (define matrices) [s]'] = np.round(T['build_matrices'], 6)
        OUT['LP (solve) [s]'] = np.round(T['solve_LP'], 6)
        
        if args.num_deriv > 1:
            OUT['Max. derivatives'] = list(np.round(deriv['LP'], 6))
            OUT['Argmax. derivatives'] = [p.name for p in pmc.parameters[ deriv['LP_idxs']]]
            OUT['Max. validation'] = list(np.round(deriv['validate'], 6))
            OUT['Difference %'] = list(np.round(deriv['RelDiff'], 6))
            
        else:
            OUT['Max. derivatives'] = np.round(deriv['LP'][0], 6)
            OUT['Argmax. derivatives'] = pmc.parameters[ deriv['LP_idxs']][0].name
            OUT['Max. validation'] = np.round(deriv['validate'][0], 6)
            OUT['Difference %'] = np.round(deriv['RelDiff'][0], 6)
        
        output_path = Path(args.root_dir, args.output_folder)
        if not os.path.exists(output_path):
        
           # Create a new directory because it does not exist
           os.makedirs(output_path)
        
        dt = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
        out_file = os.path.join(args.output_folder, Path(args.model).stem + dt + '.json')
        with open(str(Path(args.root_dir, out_file)), "w") as outfile:
            json.dump(OUT, outfile)