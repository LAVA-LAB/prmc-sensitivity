# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:44:50 2022

@author: Thom Badings
"""

import numpy as np
import cvxpy as cp
from tabulate import tabulate

import stormpy

from models.uncertainty_models import L0_polytope, L1_polytope

class prMDP:
    
    def __init__(self, num_states):
        
        self.states_dict = {}
        self.parameters = {}
        self.parameters_max_value = {}
        self.param2stateAction = {}
        
        self.robust_pairs_suc = {}
        self.robust_constraints = 0
        self.robust_successors = 0
        
        # Adjacency matrix between successor states and polytope constraints
        self.poly_pre_state = {s: set() for s in range(num_states)}
        self.distr_pre_state = {s: set() for s in range(num_states)}
        
    def __str__(self):
        
        items = {
            'No. states': len(self.states),
            'No. terminal states': len(self.states_term),
            'No. parameters': len(self.parameters),
            'Robust transitions': self.robust_successors,
            'Robust constraints': self.robust_constraints,
            'Discount factor': self.discount
            }
        
        print_list = [[k,v] for (k,v) in items.items()]
        
        return '\n' + tabulate(print_list, headers=["Property", "Value"]) + '\n'
        
    def set_state_iterator(self):
        
        self.states = self.states_dict.values()
        
    def set_initial_states(self, sI):
        
        self.sI = {'s': np.array(sI), 'p': np.full(len(sI), 1/len(sI))}
        
    def set_nonterminal_states(self):
        
        self.states_term = set([s.id for s in self.states if s.terminal])
        self.states_nonterm = self.get_state_set() - self.states_term

    def set_reward_models(self, rewards):
        
        self.rewards = rewards
        
    def get_state_set(self):
        
        return set(self.states_dict.keys())

class state:
    
    def __init__(self, id):
        
        self.id = id
        self.initial = False
        self.actions_dict = {}

    def set_action_iterator(self):
        
        self.actions = self.actions_dict.values()
        
class action:
    
    def __init__(self, id):
        
        self.id = id
        self.model = None # Uncertainty model
        self.deterministic = False # Is this transition deterministic?
        self.robust = False # Action has uncertain/robust probabilities?
        self.successors = []
        
class distribution:
    
    def __init__(self, states, probabilities):
        
        self.states = states
        self.probabilities = probabilities
        
class polytope:
    
    def __init__(self, A, b):
        
        self.A = A
        self.b = b

def parse_prmdp(path, formula, policy, args, verbose = False):

    print('Load PRISM model with STORM...')
    
    program = stormpy.parse_prism_program(path)
    model = stormpy.build_model(program)

    if model.is_nondeterministic_model and not formula is None and policy == 'optimal':
        
        formulas = stormpy.parse_properties(formula, program)
        
        result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)
        scheduler = result.scheduler

    else:
        scheduler = None
    
    #####
    
    # Set uncertainty model and its size
    if args.uncertainty_model == 'L0':
        uncertainty_model = L0_polytope
    else:
        uncertainty_model = L1_polytope
    norm_size = args.uncertainty_size
    
    print('Parsing PRISM model...')    

    # Minimum margin between [0,1] bounds and every transition probability
    DELTA = 1e-6

    # Define graph object
    M = prMDP(num_states = len(model.states))
    M.set_initial_states(model.initial_states)
    
    M.discount = args.discount
    M.beta_penalty = args.beta_penalty
    
    for s in model.states:
        
        if verbose:
            print('\n- Add state {}, labels {}'.format(s.id, s.labels))
        M.states_dict[s.id] = state(s.id)
        
        # If current state has one of the terminal labels, make this state
        # a terminal state
        if any([label in args.terminal_label for label in s.labels]):
            M.states_dict[s.id].terminal = True
            if verbose:
                print('-- States {} is terminal'.format(s.id))
        
        else:
            
            # Check if this state is an initial state
            if s.id in model.initial_states:
                if verbose:
                    print('-- State {} is an initial state'.format(s.id))
                M.states_dict[s.id].initial = True
            
            M.states_dict[s.id].terminal = False
                
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
                probabilities   = np.array([t.value() for t in a.transitions])
                
                M.states_dict[s.id].actions_dict[a.id].successors = successors
                
                if len(successors) == 1:
                    
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
                    
                    # Determine margin between precise distribution and simplex
                    norm_concat = np.concatenate([probabilities - DELTA, 1 - probabilities + DELTA])
                    max_value = np.min(np.abs(norm_concat))
                    par_value   = min(max_value, norm_size)
                    
                    if norm_size >= max_value:
                        print(' -- Reduce size of L1 uncertainty set to', max_value)
                    M.parameters[(s.id, a.id)] = cp.Parameter(value = par_value)
                    M.parameters_max_value[(s.id, a.id)] = max_value
                    
                    # Keep track of to which state-action pairs each parameter belongs
                    M.param2stateAction[ M.parameters[(s.id, a.id)] ] = [(s.id, a.id)]
                    
                    A, b = uncertainty_model(probabilities, M.parameters[(s.id, a.id)])
                    
                    M.states_dict[s.id].actions_dict[a.id].model = polytope(A, b)

                    # Keep track of all robust state-action pairs
                    M.robust_pairs_suc[(s.id, a.id)] = len(successors)
                    M.robust_successors += len(successors)
                    
                    # Put an (arbitrary) ordering over the dual variables
                    M.states_dict[s.id].actions_dict[a.id].alpha_start_idx = M.robust_constraints
                    M.robust_constraints += len(b)
                    
                    
        # Set action iterator
        M.states_dict[s.id].set_action_iterator()

    # set state iterator
    M.set_state_iterator()

    # Add reward model
    reward_model = next(iter(model.reward_models.values()))
    M.set_reward_models(reward_model)
    M.set_nonterminal_states()
    
    # Give an index to every parameter
    M.par_idx2tuple = list(M.parameters.keys())
    
    # Determine which parameters are already at their maximum value
    M.pars_at_max = M.pars_at_max = np.array([True if v.value >= M.parameters_max_value[(s,a)] else False for (s,a),v in M.parameters.items()])
    M.param_index  = np.arange(len(M.parameters))
    
    return M

'''
def parse_policy(M, policy, verbose = False):
    
    print('Parsing policy...')
    
    # Policy is None means that the model is not nondeterministic
    if policy is None:
        nondet = False
    else:
        nondet = True
      
    PI = {}  
      
    # Iterate over all state-action pairs
    for s in M.states:
        if s.id not in M.states_term:
            if verbose:
                print('- Parse policy for state {}'.format(s.id))
            
            # If the model is not nondeterministic, choose action zero
            if not nondet:
                PI[s.id] = {s.actions_dict[0]: 1}
                
            # Otherwise, follow provided policy
            else:
                choice = policy.get_choice(s.id)
                action = choice.get_deterministic_choice()
                
                PI[s.id] = {s.actions_dict[action]: 1}
    
    return PI

def generate_random_policy(M, verbose = False):
    
    PI = {}
    
    for s in M.states:
        if s.id not in M.states_term:
            num_actions = len(s.actions)
            PI[s.id] = {a: 1/num_actions for a in s.actions}
            
    return PI
'''