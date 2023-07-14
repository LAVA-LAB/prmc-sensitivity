import numpy as np
import cvxpy as cp

from core.uncertainty_models import Linf_polytope, L1_polytope, Hoeffding_interval
from core.classes import PRMC, state, action, distribution, polytope
from core.sensitivity import gradient, solve_cvx_gurobi
from core.verify_prmc import verify_prmc
from core.io.export import export_json

from core.pmc_functions import pmc_instantiate

import sys
import time
from gurobipy import GRB

def pmc2prmc(pmc_model, pmc_parameters, pmc_scheduler, point, sample_size, args, verbose, T = False):
    '''
    Convert pMC into a prMC

    Parameters
    ----------
    pmc_model : Stormpy pMC model
    pmc_parameters : Numpy array of pMC parameters
    pmc_scheduler : Dictionary for scheduler/policy of the policy
    point : Parameter point
    sample_size : Sample size to use
    args : Arguments object
    verbose : Boolean for verbose output
    T : Timing object, used to store run times

    Returns
    -------
    M : prMC object

    '''    

    start_time = time.time()    
    
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
    
    M.is_sink_state = pmc_model.is_sink_state
    
    #TODO: has not been tested with discount < 1.
    M.discount = args.discount
    
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
                
            # Copy scheduler/policy for this state
            M.states_dict[s.id].policy = pmc_scheduler[s.id]
            
            # For all possible actions that are also in the policy...
            for a in s.actions:
                
                # Check if this action is ever chosen under the specified policy
                if M.states_dict[s.id].policy[a.id] == 0:
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
                cnd1 = len(successors) == 1 or not args.robust_probabilities[s.id]
                
                # If dependencies is set to 'parameter' level, then
                # create a precise probability distribution if there
                # are multiple or zero involved parmaeters.
                cnd2 = len(involved_pmc_parameters) != 1 and not args.no_par_dependencies
                
                if cnd1 or cnd2:
                    if verbose:
                        print('Pair ({},{}) is deterministic'.format(s.id, a.id))
                    
                    # Deterministic transition (no uncertainty model)
                    M.states_dict[s.id].actions_dict[a.id].robust = False
                    
                    M.states_dict[s.id].actions_dict[a.id].model = \
                            distribution(successors, probabilities)
                
                # Otherwise, state-action pair is robust
                else:
                    
                    # State is nonterminal and distribution is uncertain/robust
                    M.states_dict[s.id].actions_dict[a.id].robust = True
                    
                    M.states_dict[s.id].actions_dict[a.id].type = uncertainty_model
                    
                    if not args.no_par_dependencies and len(involved_pmc_parameters) == 1:
                        # If there's only one involved parameter, associate a single sample size with it in every (s,a) pair
                        v = list(involved_pmc_parameters)[0]
                        if v not in M.parameters:
                            M.parameters[v] = cp.Parameter(value = sample_size[v.name])
                        
                    else:
                        # Create a new sample size parameter for each (s,a) pair
                        v = str((s.id, a.id))
                        M.parameters[v] = cp.Parameter(value = args.default_sample_size)
                        
                    if verbose:
                        print('Create parameter {} for ({},{})'.format(v, s.id, a.id))

                    M.states_dict[s.id].actions_dict[a.id].storm_parameter = v

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
                    M.robust_successors[(s.id, a.id)] = len(successors)
                    
        # Set action iterator
        M.states_dict[s.id].set_action_iterator()

    # set state iterator
    M.set_state_iterator()
    
    # Set an ordering over the robust constraints
    M.set_robust_constraints()
    
    # Assign an index to every prMC parameter
    M.paramIndex = np.array(list(M.parameters.keys()))
    
    if T:
        T.times['initialize_model'] = time.time() - start_time
    
    return M


def prmc_verify(prmc, pmc, args, verbose, T = False):
    '''
    Verify prMC

    Parameters
    ----------
    prmc : prMC object
    pmc : pMC object
    args : Arguments object
    verbose : Boolean for verbose output
    T : Timing object, used to store run times

    Returns
    -------
    P : Verifier object
    solution : Obtained solution (scalar)

    '''
    
    solver_verbose = args.verbose

    start_time = time.time()
    P = verify_prmc(prmc, pmc.reward, args.beta_penalty, args.robust_bound, verbose = verbose)
    print('- Solve optimization problem...')
    
    P.cvx.Params.NumericFocus = 3
    P.cvx.Params.ScaleFlag = 1
    
    P.solve(store_initial = True, verbose=solver_verbose)
    
    if T:
        T.times['verify_model'] = time.time() - start_time
       
    print('Range of solutions: [{}, {}]'.format(np.min(P.x_tilde), np.max(P.x_tilde)))
    print('Solution in initial state: {}\n'.format(P.x_tilde[pmc.sI['s']] @ pmc.sI['p']))
    
    print('Check complementary slackness...')
    
    violated = P.get_active_constraints(prmc, verbose = False)
    if violated:
        print(  "\n--------------------------------------------------------------\n"
                "WARNING: The Slackness conditions of the LP solution are violated.\n\n"
                "This is caused by an inproper nr. of active constraints in the LP for the prMC.\n"
                "I will try to proceed for now, but this warning can mean that the derivative is\n"
                "not defined, in which case you can encounter an error later in the program.\n\n"
                "For details, see Section 5.2 of our paper: [Badings et al., 2023, CAV].\n"
                "--------------------------------------------------------------")
    else:
        print('- Slackness conditions satisfied, proceed')
        
    solution = P.x_tilde[pmc.sI['s']] @ pmc.sI['p']
            
    return P, solution


def prmc_derivative_LP(prmc, pmc, P, args, T = False):
    '''
    Compute k=args.num_deriv derivatives for prMC

    Parameters
    ----------
    prmc : prMC object
    pmc : pMC object
    P : Verifier object
    args : Arguments object
    T : Timing object, used to store run times
    
    Returns
    -------
    G : Derivatives object
    deriv : Dictionary of derivative results

    '''
    
    # Create object for computing gradients
    start_time = time.time()
    G = gradient(prmc, args.robust_bound)
    
    # Update gradient object with current solution
    G.update_LHS(prmc, P)
    if T:
        T.times['compute_LHS_matrix'] = time.time() - start_time
        
    start_time = time.time()
    G.update_RHS(prmc, P)
    if T:
        T.times['compute_RHS_matrix'] = time.time() - start_time
    
    print('Compute parameter importance via LP (GurobiPy)...')
    start_time = time.time()
    
    deriv = {}
    
    if args.robust_bound == 'lower':
        direction = GRB.MAXIMIZE
    else:
        direction = GRB.MINIMIZE
        
    print('- Shape of J matrix:', G.J.shape, G.Ju.shape)
    
    if args.num_deriv > G.Ju.shape[1]:
        raise ValueError("Abort, because the number of requested derivative is higher than the number of parameters.")
    
    optm, deriv['LP_idxs'], deriv['LP'] = solve_cvx_gurobi(G.J, G.Ju, pmc.sI, args.num_deriv,
                                direction=direction, verbose=args.verbose)
    
    deriv['LP_pars'] = np.array([prmc.parameters[sa] for sa in prmc.paramIndex[deriv['LP_idxs']]])
    
    for i,par in enumerate(deriv['LP_pars']):
        deriv['LP_pars'] = par.name()
        
    if T:
        T.times['solve_k_highest_derivatives'] = time.time() - start_time   
        print('- LP solved in: {:.3f} sec.'.format(T.times['solve_k_highest_derivatives']))
    print('- Obtained derivatives are {} for parameters {}'.format(deriv['LP'],  deriv['LP_pars']))
            
    return G, deriv


def prmc_validate_derivative(prmc, pmc, inst, solution, deriv, delta, 
                             robust_bound, beta_penalty, verbose = False):
    '''
    Validate derivatives numerically, by giving each parameter a small delta
    and recomputing the solution

    Parameters
    ----------
    prmc : prMC object
    pmc : pMC object
    inst : Instantiation dictionary
    solution : Current solution (scalar)
    deriv : Dictionary of derivative results
    args : Arguments object
    verbose : Boolean for verbose output
    
    
    pmc : pMC object
    inst : Parameter instantiation dictionary
    solution : Current solution (scalar value)
    deriv : Dictionary with derivative results
    delta : delta to give to the parameter instantiation

    Returns
    -------
    deriv : Updated derivatives object

    '''
    
    print('\nValidation by perturbing parameters by +{}'.format(delta))
    
    deriv['validate'] = np.zeros(len(deriv['LP_idxs']), dtype=float)
    deriv['RelDiff']  = np.zeros(len(deriv['LP_idxs']), dtype=float)
    
    for q,x in enumerate(prmc.paramIndex[deriv['LP_idxs']]):
        
        print('- Check parameter {}'.format(x))
        prmc.parameters[x].value += delta
        
        P = verify_prmc(prmc, pmc.reward, beta_penalty, robust_bound, verbose = verbose)
        
        P.cvx.Params.NumericFocus = 3
        P.cvx.Params.ScaleFlag = 1
        
        P.solve(store_initial = True, verbose=verbose)
        
        solution_new = P.x_tilde[pmc.sI['s']] @ pmc.sI['p']
        
        # Compute derivative
        deriv['validate'][q] = (solution_new-solution) / delta
        
        # Determine difference in %
        if deriv['LP'][q] == 0:
            if deriv['validate'][q] != 0:
                deriv['RelDiff'][q] = np.inf
        else:
            deriv['RelDiff'][q] = (deriv['validate'][q]-deriv['LP'][q])/deriv['LP'][q]
        
        print('- Parameter {}, LP: {:.6f}, val: {:.6f}, diff: {:.6f}'.format(x, deriv['LP'][q], deriv['validate'][q], deriv['RelDiff'][q]))
        
        prmc.parameters[x].value -= delta
        
    return deriv