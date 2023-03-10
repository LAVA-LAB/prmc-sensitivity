from core.baseline_gradient import explicit_gradient

import numpy as np
import scipy.sparse as sparse
import time
import stormpy
import sys

import gurobipy as gp
from gurobipy import GRB


def pmc_get_reward(pmc, model, args):
    '''
    Get the reward vector for the given pMC.
    '''
    
    # If nondeterministic model, first use storm to obtain a policy
    if model.is_nondeterministic_model:
        
        print('- Compute policy to resolve nondeterminism')
        
        storm_solution, pmc.scheduler_raw = verify_pmdp_storm(model, pmc.properties)
        pmc.scheduler_prob = convert_scheduler(model, pmc.scheduler_raw)
    
    else:
        
        storm_solution = verify_pmc_storm(model, pmc.properties)
        pmc.scheduler_prob = {s.id: np.array([1]) for s in model.states}
        
    print('\n(Storm) range of solutions: [{}, {}]'.format(np.min(storm_solution), np.max(storm_solution)))
    print('(Storm) Solution in initial state: {}'.format(storm_solution[pmc.sI['s']] @ pmc.sI['p']))
    
    print('\nSet reward vector')
    
    if args.goal_label is not None:
        reward = get_pmdp_reward_from_label(model, args.goal_label)
    else:
        reward = get_pmdp_reward_vector(model, pmc.scheduler_prob)
        
    print('')
        
    return reward
        


def get_pmdp_reward_from_label(model, label):
    '''
    Set reward vector for reachability problem (rew=1 for goal states only).
    '''
    
    print('- Set R=1 for states with label "{}"'.format(label))
    
    reward = np.zeros(len(model.states))
    all_labels = set({})
    
    nr_mod = 0
    for s in model.states:
        all_labels.update(model.labels_state(s.id))
        if label.issubset(model.labels_state(s.id)):
            reward[s.id] = 1
            nr_mod += 1
            
    print('- Set reward to one for {} states'.format(nr_mod))
    print('- All encountered labels are:', all_labels)
    
    return reward



def get_pmdp_reward_vector(model, scheduler):
    
    print('- Load reward model')
    
    if len(model.reward_models) == 0:
        raise Exception("No reward model specified.")
    
    reward_model = next(iter(model.reward_models.values()))
    reward = np.zeros(len(model.states))
    
    for s in model.states:
        
        if not model.is_sink_state(s):
            if reward_model.has_state_rewards:
                reward[s.id] = float(reward_model.get_state_reward(s.id))
                
            elif reward_model.has_state_action_rewards:
                # For each action
                for action,prob in enumerate(scheduler[s.id]):
                    
                    # Check if model is nondeterministic
                    if model.is_nondeterministic_model:
                        # Get choice ID of state-action pair
                        choice = model.get_choice_index(s, action)
                    else:
                        # Otherwise, choice ID = state ID
                        choice = s.id
                        
                    # Reward is weighted with probability of chosing that action
                    reward[s.id] += prob * float(reward_model.get_state_action_reward(choice))
                                    
            else:
                sys.exit()
                
    return reward



def pmc_verify(instantiated_model, pmc, point, T = False):
    '''
    Compute the solution for the given instantiated model.

    Parameters
    ----------
    instantiated_model : Stormpy instantiated model
    pmc : pMC object
    point : Parameter point
    T : Timing object, used to store run times

    Returns
    -------
    solution_sI : Solution in initial state (scalar)
    J : LHS matrix
    Ju : RHS matrix

    '''
    
    print('Model checking pMC...')
        
    start_time = time.time()
    
    # After that, use spsolve to obtain the actual solution    
    J, result = verify_linear_program(instantiated_model, pmc.reward, pmc.scheduler_prob)
    #J, result = verify_spsolve(instantiated_model, pmc.reward, pmc.scheduler_prob)
    
    if T:
        if not 'verify' in T.times:
            T.times['verify'] = time.time() - start_time
    
    print('Range of solutions: [{}, {}]'.format(np.min(result), np.max(result)))
    print('Solution in initial state: {}\n'.format(result[pmc.sI['s']] @ pmc.sI['p']))
        
    start_time = time.time()
    Ju = define_sparse_RHS(pmc.model, pmc.parameters, pmc.get_parameters_to_states(), result, point, pmc.scheduler_prob)
    
    if T:
        T.times['build_matrices'] = time.time() - start_time

    # Retrieve actual solution
    solution_sI = result[pmc.sI['s']] @ pmc.sI['p']

    return solution_sI, J, Ju



def convert_scheduler(model, scheduler):
    '''
    Convert deterministic scheduler to probability distribution over actions 
    in each state.
    '''
    
    prob_scheduler = {}
    
    for state in model.states:
        
        # Initialize to all actions zero probability
        prob_scheduler[state.id] = np.zeros(len(state.actions))

        # Get deterministic choice from scheduler
        choice = scheduler.get_choice(state)
        
        action = choice.get_deterministic_choice()
        
        # Set the action chosen by the scheduler to probability of one
        prob_scheduler[state.id][action] = 1
        
    return prob_scheduler
        


def verify_spsolve(model, reward, scheduler):

    print('- Verify by solving sparse equation system...')    
    
    J = define_sparse_LHS(model, scheduler)
    result = sparse.linalg.spsolve(J, reward)  
        
    return J, result



def verify_linear_program(model, reward, scheduler, direction = GRB.MINIMIZE):
    
    print('- Verify by solving LP using Gurobi')
    
    J = define_sparse_LHS(model, scheduler)
    
    # Define LP
    m = gp.Model('CVX')
    
    # m.Params.NumericFocus = 3
    # m.Params.ScaleFlag = 1
    
    print('--- Define optimization model...')
    
    # Add reward veriables
    result = m.addMVar(J.shape[1], lb=0, ub=10e6)
    
    # Add constraint
    m.addConstr(J @ result == reward)

    # Define objective
    m.setObjective(gp.quicksum(result), direction)
    
    m.Params.Method = 2
    m.Params.Crossover = 0
    
    print('--- Solve...')
    
    # Solve
    m.optimize()
    
    return J, result.X



def verify_pmc_storm(model, properties):
    
    # Compute model checking result
    result = stormpy.model_checking(model, properties[0])
    array  = np.array(result.get_values(), dtype=float)

    return array



def verify_pmdp_storm(model, properties):
    
    # Compute model checking result
    result = stormpy.model_checking(model, properties[0], extract_scheduler=True)
    array  = np.array(result.get_values(), dtype=float)
    
    assert result.has_scheduler
    scheduler = result.scheduler
    assert scheduler.memoryless
    assert scheduler.deterministic
    
    return array, scheduler
    


def define_sparse_LHS(model, scheduler):
    '''
    Construct the left-hand side matrix for computing derivatives.
    '''
    
    row = []
    col = []
    val = []
    
    z = 0
    dok = {}
    
    for state in model.states:
        for action in state.actions:
            # Only proceed if this action is chosen by the policy
            if scheduler[state.id][action.id] == 0:
                continue
            
            for transition in action.transitions:
                
                if not model.is_sink_state(state.id):
                
                    # Retrieve policy
                    pi = scheduler[state.id][action.id]
                
                    value = pi * float(transition.value())
                    # value = pi * float(transition.value().evaluate(subpoint[ID]))
                    
                    if value != 0:
                
                        # Add or update entry
                        if (state.id, transition.column) not in dok:
                            # Add to sparse matrix
                            row += [state.id]
                            col += [transition.column]
                            val += [value]
                            
                            dok[(state.id, transition.column)] = z
                            z += 1
                            
                        else:
                            val[dok[(state.id, transition.column)]] += value
        
    # Create sparse matrix for left-hand side
    J = sparse.identity(len(model.states)) - sparse.csc_matrix((val, (row, col)), shape=(len(model.states), len(model.states)))
    
    for state in model.states:
        if not model.is_sink_state(state.id):
            error = np.sum(J[state.id,:])
            if error != 0:
                J[state.id, state.id] -= error
                
        else:
            error = np.sum(J[state.id,:])
            if error != 1:
                assert False
    
    return J
    


def define_sparse_RHS(model, parameters, params2states, sols, point, scheduler):
    '''
    Construct the right-hand side matrix for computing derivatives.
    '''
    
    row = []
    col = []
    val = []
    
    z = 0
    dok = {}
    
    for q,p in enumerate(parameters):
        
        for state in params2states[p]:
            
            # If the value in this state is zero, we can skip it anyways
            if sols[state.id] != 0:        
                for action in state.actions:
                    
                    # Only proceed if this action is chosen by the policy
                    if not scheduler[state.id][action.id] == 1:
                        continue
                    
                    cur_val = 0
                    
                    for transition in action.transitions:
                        
                        # Gather variables related to this transition
                        var = transition.value().gather_variables()
                        
                        # Get valuation for only those parameters
                        subpoint = {v: point[v] for v in var}
                        
                        # Retrieve policy
                        pi = scheduler[state.id][action.id]
                        
                        value = pi * float(transition.value().derive(p).evaluate(subpoint))
                        
                        cur_val += value * sols[transition.column]
                        
                    if cur_val != 0:
                
                        # Add or update entry
                        if (state.id, q) not in dok:
                            # Add to sparse matrix
                            row += [state.id]
                            col += [q]
                            val += [cur_val]
                            
                            dok[(state.id, q)] = z
                            z += 1
                            
                        else:
                            val[dok[(state.id, q)]] += cur_val
                        
    # Create sparse matrix for right-hand side
    Ju = -sparse.csc_matrix((val, (row, col)), shape=(len(model.states), len(parameters)))
    
    return Ju