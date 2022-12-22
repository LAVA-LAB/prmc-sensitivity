# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:44:50 2022

@author: Thom Badings
"""

import numpy as np
import cvxpy as cp

class prMDP:
    
    def __init__(self, num_states):
        
        self.states_dict = {}
        self.parameters = {}
        
        # Adjacency matrix between successor states and polytope constraints
        self.poly_pre_state = {s: set() for s in range(num_states)}
        self.distr_pre_state = {s: set() for s in range(num_states)}
        
    def set_state_iterator(self):
        
        self.states = self.states_dict.values()
        
    def set_initial_states(self, sI):
        
        if type(sI) == int:
            self.sI = {sI: 1}
        elif type(sI) == dict:
            self.sI = sI
        else:
            assert False
        
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

def parse_storm(model, policy, uncertainty_model, norm_bound, 
                terminal_labels = [], verbose = False):

    print('Parsing PRISM model...')    

    # Define graph object
    M = prMDP(num_states = len(model.states))
    M.set_initial_states(int(model.initial_states[0]))
    
    for s in model.states:
        
        if verbose:
            print('\n- Add state {}, labels {}'.format(s.id, s.labels))
        M.states_dict[s.id] = state(s.id)
        
        # If current state has one of the terminal labels, make this state
        # a terminal state
        if any([label in terminal_labels for label in s.labels]):
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
                
            # For all possible actions
            for a in s.actions:
                if verbose:
                    print('-- Add action {} for state {} with {} transitions'.format(a.id, s.id, len(a.transitions)))
                M.states_dict[s.id].actions_dict[a.id] = action(a.id)
            
                successors      = np.array([t.column for t in a.transitions])
                probabilities   = np.array([t.value() for t in a.transitions])
                
                M.states_dict[s.id].actions_dict[a.id].successors = successors
                
                if len(successors) == 1 or s.id == 0:
                    
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
                    norm_concat = np.concatenate([probabilities, 1-probabilities])
                    norm_margin = np.min(np.abs(norm_concat))
    
                    if norm_bound > norm_margin and verbose:
                            print(' -- Reduce size of L1 uncertainty set to', norm_margin)
                    M.parameters[(s.id, a.id)] = cp.Parameter(value = min(norm_margin, norm_bound))
                    
                    A, b = uncertainty_model(probabilities, M.parameters[(s.id, a.id)])
                    
                    M.states_dict[s.id].actions_dict[a.id].model = polytope(A, b)

        # Set action iterator
        M.states_dict[s.id].set_action_iterator()

    # set state iterator
    M.set_state_iterator()

    # Add reward model
    reward_model = next(iter(model.reward_models.values()))
    M.set_reward_models(reward_model)
    M.set_nonterminal_states()
    
    return M

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