import numpy as np
import cvxpy as cp

from core.uncertainty_models import Linf_polytope, L1_polytope, Hoeffding_interval
from core.classes import PRMC, state, action, distribution, polytope
from core.sensitivity import gradient, solve_cvx_gurobi
from core.cvx_verification_prmc import cvx_verification_gurobi
from core.baseline_gradient import explicit_gradient
from core.export import export_json

from core.pmc_functions import pmc_instantiate

import sys
import time
from gurobipy import GRB

def pmc2prmc(pmc_model, pmc_parameters, point, sample_size, args, verbose, T = False):

    start_time = time.time()    

    scheduler = None
    
    #####
    
    # Set uncertainty model and its size
    if args.uncertainty_model == "Hoeffding":
        uncertainty_model = Hoeffding_interval
    elif args.uncertainty_model == 'Linf':
        uncertainty_model = Linf_polytope
    else:
        uncertainty_model = L1_polytope

    # Define graph object
    M = PRMC(num_states = len(pmc_model.states))
    M.set_initial_states(pmc_model.initial_states)
    
    M.discount = args.discount
    M.beta_penalty = args.beta_penalty
    
    M.is_sink_state = pmc_model.is_sink_state
    
    M.nr_transitions = pmc_model.nr_transitions
    M.parameters_pmc = pmc_parameters
    
    # Parse model
    for s in pmc_model.states:
        
        if verbose:
            print('\n- Add state {}, labels {}'.format(s.id, s.labels))
        M.states_dict[s.id] = state(s.id)
        
        # Only proceed if not a sink state
        if not pmc_model.is_sink_state(s.id):
            
            # Check if this state is an initial state
            if s.id in pmc_model.initial_states:
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
                
                # Check if this action is ever chosen under the specified policy
                if a.id not in M.states_dict[s.id].policy.keys():
                    continue
                
                if verbose:
                    print('-- Add action {} for state {} with {} transitions'.format(a.id, s.id, len(a.transitions)))
                
                # Create an action
                M.states_dict[s.id].actions_dict[a.id] = action(a.id)
                
                # Retrieve successor states
                successors      = np.array([t.column for t in a.transitions])
                M.states_dict[s.id].actions_dict[a.id].successors = successors
                M.states_dict[s.id].actions_dict[a.id].parametricTrans = a.transitions
                
                # Retrieve precise probabilities
                probabilities   = np.array([float(t.value().evaluate(point)) for t in a.transitions])
                
                # Retrieve involved parameters
                involved_pmc_parameters = set({})
                for t in a.transitions:
                    var = t.value().gather_variables()
                    involved_pmc_parameters.update(var)
                
                # If only one successor states, or if the distribution in this state-action pair is not robust
                if len(successors) == 1 or not args.robust_probabilities[s.id]:
                    print('Pair ({},{}) is deterministic'.format(s.id, a.id))
                    
                    # Deterministic transition (no uncertainty model)
                    M.states_dict[s.id].actions_dict[a.id].deterministic = True
                    M.states_dict[s.id].actions_dict[a.id].robust = False
                    
                    M.states_dict[s.id].actions_dict[a.id].model = \
                            distribution(successors, probabilities)
                
                # Otherwise, state-action pair is robust
                else:
                    
                    # State is nonterminal and distribution is uncertain/robust
                    M.states_dict[s.id].actions_dict[a.id].deterministic = False
                    M.states_dict[s.id].actions_dict[a.id].robust = True
                    
                    M.stateAction2param[(s.id, a.id)] = list(involved_pmc_parameters)
                    
                    M.states_dict[s.id].actions_dict[a.id].type = uncertainty_model
                    
                    if args.robust_dependencies == 'parameter' and len(involved_pmc_parameters) == 1:
                        # If there's only one involved parameter, associate a single sample size with it in every (s,a) pair
                        v = list(involved_pmc_parameters)[0]
                        if v not in M.parameters:
                            M.parameters[v] = cp.Parameter(value = sample_size[v.name])
                        
                    else:
                        print('Create parameter for ({},{})'.format(s.id, a.id))
                        
                        # Create a new sample size parameter for each (s,a) pair
                        v = str((s.id, a.id))
                        M.parameters[v] = cp.Parameter(value = args.default_sample_size)
                        
                    w = M.parameters[v]
                        
                    # Keep track of to which state-action pairs each parameter belongs
                    if v in M.param2stateAction:
                        M.param2stateAction[v] += [(s.id, a.id)]
                    else:
                        M.param2stateAction[v] = [(s.id, a.id)]
                    
                    if uncertainty_model == Hoeffding_interval:
                        # If Hoeffding's based uncertainty set is used, also provide confidence level
                        A, b = uncertainty_model(probabilities, args.robust_confidence, w)
                        
                    else:
                        # Otherwise, directly give the center and size of the uncertainty set (L1, Linf, ...)
                        A, b = uncertainty_model(probabilities, 0.01)
                    
                    M.states_dict[s.id].actions_dict[a.id].model = polytope(A, b, uncertainty_model, w, args.robust_confidence)

                    # Keep track of all robust state-action pairs
                    M.robust_pairs_suc[(s.id, a.id)] = len(successors)
                    M.robust_successors += len(successors)
                    
        # Set action iterator
        M.states_dict[s.id].set_action_iterator()

    # set state iterator
    M.set_state_iterator()
    
    # Set an ordering over the robust constraints
    M.set_robust_constraints()
    
    # Assign an index to every prMC parameter
    M.paramIndex = np.array(list(M.parameters.keys()))
    
    if T:
        T.times['instantiate'] = time.time() - start_time
    
    return M


def prmc_verify(prmc, pmc, args, verbose, T = False):
    
    done = False
    trials = 0
    trials_max = 1
    solver_verbose = args.verbose
    while not done:

        start_time = time.time()
        P = cvx_verification_gurobi(prmc, pmc.reward, args.robust_bound, verbose = verbose)
        print('- Solve optimization problem...')
        
        P.cvx.Params.NumericFocus = 3
        P.cvx.Params.ScaleFlag = 1
        
        P.solve(store_initial = True, verbose=solver_verbose)
        
        if T:
            T.times['verify'] = time.time() - start_time
           
        print('Range of solutions: [{}, {}]'.format(np.min(P.x_tilde), np.max(P.x_tilde)))
        print('Solution in initial state: {}\n'.format(P.x_tilde[pmc.sI['s']] @ pmc.sI['p']))
        
        print('Check complementary slackness...')
        
        if not P.check_complementary_slackness(prmc, verbose=False):
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
            
    solution = P.x_tilde[pmc.sI['s']] @ pmc.sI['p']
            
    return P, solution


def prmc_derivative_LP(prmc, pmc, P, args, T = False):
    
    start_time = time.time()
    # Create object for computing gradients
    G = gradient(prmc, args.robust_bound)
    
    # Update gradient object with current solution
    G.update(prmc, P, mode='remove_dual')
    if T:
        T.times['build_matrices'] = time.time() - start_time
    
    deriv = {}
    
    print('Compute parameter importance via LP (GurobiPy)...')
    start_time = time.time()
    
    if args.robust_bound == 'lower':
        direction = GRB.MAXIMIZE
    else:
        direction = GRB.MINIMIZE
        
    print('- Shape of J matrix:', G.J.shape, G.Ju.shape)
        
    deriv['LP_idxs'], deriv['LP'] = solve_cvx_gurobi(G.J, G.Ju, pmc.sI, args.num_deriv,
                                direction=direction, verbose=args.verbose)
    
    deriv['LP_pars'] = np.array(list(prmc.parameters.values()))[ deriv['LP_idxs'] ]
    
    import cvxpy as cp
    for i,par in enumerate(deriv['LP_pars']):
        # if isinstance(par, cp.Parameter):
        deriv['LP_pars'] = par.name()
        # else:
            # deriv['LP_pars'] = par.name
    
    if T:
        T.times['solve_LP'] = time.time() - start_time   
        print('- LP solved in: {:.3f} sec.'.format(T.times['solve_LP']))
    print('- Obtained derivatives are {} for parameters {}'.format(deriv['LP'],  deriv['LP_pars']))
    
    if args.explicit_baseline:
        deriv['explicit'], = \
            explicit_gradient(pmc, args, G.J, G.Ju, T)
            
    return deriv


def prmc_validate_derivative(prmc, pmc, inst, solution, deriv, args, verbose = False):
    
    print('\nValidation by perturbing parameters by +{}'.format(args.validate_delta))
    
    deriv['validate'] = np.zeros(args.num_deriv, dtype=float)
    deriv['RelDiff']  = np.zeros(args.num_deriv, dtype=float)
    
    for q,x in enumerate(prmc.paramIndex[deriv['LP_idxs']]):
        
        args.beta_penalty = 0
        
        # prmc = pmc2prmc(pmc.model, pmc.parameters, inst['point'], inst['sample_size'], args, verbose = verbose)
        
        print('- Check parameter {}'.format(x))
        prmc.parameters[x].value += args.validate_delta
        
        P = cvx_verification_gurobi(prmc, pmc.reward, args.robust_bound, verbose=verbose)
        
        P.cvx.Params.NumericFocus = 3
        P.cvx.Params.ScaleFlag = 1
        
        P.solve(store_initial = True, verbose=verbose)
        
        solution_new = P.x_tilde[pmc.sI['s']] @ pmc.sI['p']
        
        # Compute derivative
        deriv['validate'][q] = (solution_new-solution) / args.validate_delta
        
        # Determine difference in %
        deriv['RelDiff'][q] = (deriv['validate'][q]-deriv['LP'][q])/deriv['LP'][q]
        
        print('- Parameter {}, LP: {:.6f}, val: {:.6f}, diff: {:.6f}'.format(x, deriv['LP'][q], deriv['validate'][q], deriv['RelDiff'][q]))
        
        prmc.parameters[x].value -= args.validate_delta
        
    return deriv