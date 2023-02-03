import numpy as np
import cvxpy as cp

from core.uncertainty_models import Linf_polytope, L1_polytope, Hoeffding_interval
from core.classes import PRMC, state, action, distribution, polytope
from core.sensitivity import gradient, solve_cvx_gurobi
from core.verify_prmc import cvx_verification_gurobi
from core.baseline_gradient import explicit_gradient
from core.export import export_json

import sys
import time
from gurobipy import GRB
import copy


def run_prmc(pmc, args, inst, verbose):

    T = {}    

    args.uncertainty_model = "Hoeffding"
    
    T['parse_model'] = 0.0
    
    print('Convert pMC to prMC...')
    
    start_time = time.time()
    prmc = pmc2prmc(pmc.model, pmc.parameters, inst['point'], inst['sample_size'], args, verbose = verbose)
    T['instantiate'] = time.time() - start_time
    
    print('Verify prMC...')
    
    # rew_backup = copy.copy(pmc.reward)
    
    done = False
    trials = 0
    trials_max = 3
    solver_verbose = args.verbose
    while not done:
    
        start_time = time.time()
        CVX_GRB = cvx_verification_gurobi(prmc, pmc.reward, args.robust_bound, verbose = verbose)
        print('- Solve optimization problem...')
        
        CVX_GRB.cvx.Params.NumericFocus = 3
        CVX_GRB.cvx.Params.ScaleFlag = 1
        
        CVX_GRB.solve(store_initial = True, verbose=solver_verbose)
        T['verify'] = time.time() - start_time
           
        print('Range of solutions: [{}, {}]'.format(np.min(CVX_GRB.x_tilde), np.max(CVX_GRB.x_tilde)))
        print('Solution in initial state: {}\n'.format(CVX_GRB.x_tilde[pmc.sI['s']] @ pmc.sI['p']))
        
        print('Check complementary slackness...')
        
        if not CVX_GRB.check_complementary_slackness(prmc):
            # Check if complementary slackness is satisfied
            if trials < trials_max:
                trials += 1
                
                prmc.beta_penalty *= 100
                solver_verbose = True
                print('- Slackness not satisfied. Increase beta-penalty to {} and try {} more times...'.format(prmc.beta_penalty, trials_max-trials))
                
                print('- Add small delta to reward vector to break symmetry...')
                pmc.reward += 1e-2*np.random.rand(len(pmc.reward))
                
            else:
                print('- Slackness not satisfied. Abort...')
                done = True
                sys.exit()
            
        else:
            print('- Slackness satisfied, proceed.')
            done = True
        
    # pmc.reward = rew_backup
    
    print('\nSet up sensitivity computation...')
    
    start_time = time.time()
    # Create object for computing gradients
    G = gradient(prmc, args.robust_bound)
    
    # Update gradient object with current solution
    G.update(prmc, CVX_GRB, mode='remove_dual')
    T['build_matrices'] = time.time() - start_time
    
    deriv = {}
    
    print('Compute parameter importance via LP (GurobiPy)...')
    start_time = time.time()
    
    if args.robust_bound == 'lower':
        direction = GRB.MAXIMIZE
    else:
        direction = GRB.MINIMIZE
        
    print('- Shape of J matrix:', G.J.shape, G.Ju.shape)
        
    deriv['LP_idxs'], deriv['LP'] = solve_cvx_gurobi(G.J, G.Ju, pmc.sI, args.num_deriv,
                                direction=direction, verbose=verbose)
    
    deriv['LP_pars'] = np.array(list(prmc.parameters.values()))[ deriv['LP_idxs'] ]
    
    import cvxpy as cp
    for i,par in enumerate(deriv['LP_pars']):
        if isinstance(par, cp.Parameter):
            deriv['LP_pars'] = par.name()
        else:
            deriv['LP_pars'] = par.name
    
    T['solve_LP'] = time.time() - start_time   
    print('- LP solved in: {:.3f} sec.'.format(T['solve_LP']))
    print('- Obtained derivatives are {} for parameters {}'.format(deriv['LP'],  deriv['LP_pars']))
    
    if args.explicit_baseline:
        deriv['explicit'], T['solve_explicit_one'], T['solve_explicit_all'] = \
            explicit_gradient(pmc, args, G.J, G.Ju)

    # Empirical validation of gradients
    solution = CVX_GRB.x_tilde[pmc.sI['s']] @ pmc.sI['p']
    
    if not args.no_gradient_validation:

        print('\nValidation by perturbing parameters by +{}'.format(args.validate_delta))
        
        deriv['validate'] = np.zeros(args.num_deriv, dtype=float)
        deriv['RelDiff']  = np.zeros(args.num_deriv, dtype=float)
        
        for q,x in enumerate(pmc.parameters[deriv['LP_idxs']]):
            
            inst['sample_size'][x.name] += args.validate_delta
            
            args.beta_penalty = 0
            
            prmc = pmc2prmc(pmc.model, pmc.parameters, inst['point'], inst['sample_size'], args, verbose = verbose)
            
            # Increment this parameter by the given delta
            inst['valuation'][x.name] += args.validate_delta
            
            # instantiate model
            instantiated_model, point = pmc.instantiate(inst['valuation'])
            
            # print(pmc.reward)
            # assert False
            
            CVX_GRB = cvx_verification_gurobi(prmc, pmc.reward, args.robust_bound, verbose=verbose)
            
            CVX_GRB.cvx.Params.NumericFocus = 3
            CVX_GRB.cvx.Params.ScaleFlag = 1
            
            CVX_GRB.solve(store_initial = True, verbose=True)
            
            solution_new = CVX_GRB.x_tilde[pmc.sI['s']] @ pmc.sI['p']
            
            # Compute derivative
            deriv['validate'][q] = (solution_new-solution) / args.validate_delta
            
            # Determine difference in %
            deriv['RelDiff'][q] = (deriv['validate'][q]-deriv['LP'][q])/deriv['LP'][q]
            
            print('- Parameter {}, LP: {:.3f}, val: {:.3f}, diff: {:.3f}'.format(x,  deriv['validate'][q], deriv['LP'][q], deriv['RelDiff'][q]))
            
            inst['sample_size'][x.name] -= args.validate_delta
            
    if not args.no_export:
        export_json(args, prmc, T, inst, solution, deriv, parameters = pmc.parameters)
            
    return prmc, T, inst, solution, deriv
    
def pmc2prmc(model, parameters, point, sample_size, args, verbose):

    scheduler = None
    
    #####
    
    # Set uncertainty model and its size
    if args.uncertainty_model == "Hoeffding":
        uncertainty_model = Hoeffding_interval
    elif args.uncertainty_model == 'Linf':
        uncertainty_model = Linf_polytope
    else:
        uncertainty_model = L1_polytope

    # Minimum margin between [0,1] bounds and every transition probability
    MIN_PROBABILITY = 1e-6

    # Define graph object
    M = PRMC(num_states = len(model.states))
    M.set_initial_states(model.initial_states)
    
    M.discount = args.discount
    M.beta_penalty = args.beta_penalty
    
    M.is_sink_state = model.is_sink_state
    
    M.nr_transitions = model.nr_transitions
    M.parameters_pmc = parameters    
    
    # Define parameters used in uncertainty sets
    for q,k in enumerate(parameters):
        
        M.parameters[k] = cp.Parameter(value = sample_size[k.name])
    
    # Parse model
    for s in model.states:
        
        if verbose:
            print('\n- Add state {}, labels {}'.format(s.id, s.labels))
        M.states_dict[s.id] = state(s.id)
        
        # Only proceed if not a sink state
        if not model.is_sink_state(s.id):
            
            # Check if this state is an initial state
            if s.id in model.initial_states:
                if verbose:
                    print('-- State {} is an initial state'.format(s.id))
                M.states_dict[s.id].initial = True
                
            # Retrieve policy for this state
            # If the model is not nondeterministic or not set, choose random action
            if scheduler is None:
                
                num_actions = len(s.actions)
                M.states_dict[s.id].policy = {a.id: 1/num_actions for a in s.actions}
                
            # Otherwise, follow provided policy
            else:
                choice = scheduler.get_choice(s.id)
                act = choice.get_deterministic_choice()
                
                M.states_dict[s.id].policy = {act: 1}
            
            # For all possible actions that are also in the policy...
            for a in s.actions:
                if a.id not in M.states_dict[s.id].policy.keys():
                    continue
                
                if verbose:
                    print('-- Add action {} for state {} with {} transitions'.format(a.id, s.id, len(a.transitions)))
                
                M.states_dict[s.id].actions_dict[a.id] = action(a.id)
            
                successors      = np.array([t.column for t in a.transitions])
                probabilities   = np.array([float(t.value().evaluate(point)) for t in a.transitions])
                
                M.states_dict[s.id].actions_dict[a.id].successors = successors
                
                # Retrieve involved parameter
                involved_parameters = set({})
                for t in a.transitions:
                    var = t.value().gather_variables()
                    involved_parameters.update(var)
                
                if len(successors) == 1:
                    
                    # Deterministic transition (no uncertainty model)
                    M.states_dict[s.id].actions_dict[a.id].deterministic = True
                    
                    # Set adjacency matrix entries
                    for succ, prob in zip(successors, probabilities):
                        M.distr_pre_state[succ].add((s.id, a.id, prob))
                    
                    M.states_dict[s.id].actions_dict[a.id].model = \
                            distribution(successors, probabilities)
                
                elif len(involved_parameters) == 0 or len(involved_parameters) > 1:    
                    
                    # Deterministic transition (no uncertainty model)
                    M.states_dict[s.id].actions_dict[a.id].deterministic = True
                    
                    # Set adjacency matrix entries
                    for succ, prob in zip(successors, probabilities):
                        M.distr_pre_state[succ].add((s.id, a.id, prob))
                    
                    M.states_dict[s.id].actions_dict[a.id].model = \
                            distribution(successors, probabilities)            
                
                else:
                    
                    # State is nonterminal and distribution is uncertain/robust              
                    M.states_dict[s.id].actions_dict[a.id].robust = True
                    
                    # Set adjacency matrix entries
                    for dim,succ in enumerate(successors):
                        M.poly_pre_state[succ].add((s.id, a.id, dim))
                        
                    flag = 0
                        
                    # if len(involved_parameters) > 1:
                    #     print('ERROR: number of parameters in state-action ({},{}) bigger than one'.format(s.id, a.id))
                    #     flag = 1
                    #     M.parameters['par'+str(s.id)] = cp.Parameter(value = 0.02)
                    #     v = 'par'+str(s.id)
                    #     # assert False
                        
                    # else:
                    v = list(involved_parameters)[0]
                    
                    # Keep track of to which state-action pairs each parameter belongs
                    if v in M.param2stateAction:
                        M.param2stateAction[ v ] += [(s.id, a.id)]
                    else:
                        M.param2stateAction[ v ] = [(s.id, a.id)]
                    
                    if flag == 1:
                        A, b = Linf_polytope(probabilities, M.parameters[v])
                    elif uncertainty_model == Hoeffding_interval and flag == 0:
                        A, b = uncertainty_model(probabilities, args.robust_confidence, M.parameters[v], MIN_PROBABILITY)
                    else:
                        A, b = uncertainty_model(probabilities, M.parameters[v])
                    
                    M.states_dict[s.id].actions_dict[a.id].model = polytope(A, b)

                    # Keep track of all robust state-action pairs
                    M.robust_pairs_suc[(s.id, a.id)] = len(successors)
                    M.robust_successors += len(successors)
                    
                    # Put an (arbitrary) ordering over the dual variables
                    M.states_dict[s.id].actions_dict[a.id].alpha_start_idx = M.robust_constraints
                    M.robust_constraints += len(b)
                    
                    
        # Set action iterator
        M.states_dict[s.id].set_action_iterator()

    # Remove parameters that do not appear anywhere in the model
    delete = []
    for th in M.parameters.keys():
        if th not in M.param2stateAction:
            delete += [th]
            
    for d in delete:
        del M.parameters[d]
    
    iters = sum([len(M.param2stateAction[th]) for th in M.parameters.keys()])

    # set state iterator
    M.set_state_iterator()
    
    # Give an index to every parameter
    M.par_idx2tuple = list(M.parameters.keys())
    
    return M