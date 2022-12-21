# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:44:50 2022

@author: Thom Badings
"""

import numpy as np
import cvxpy as cp

class prMDP:
    
    def __init__(self):
        
        self.states_dict = {}
        self.parameters = {}
        
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
        self.selfloop = False # All transitions lead back to same state?
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

def parse_storm(model, uncertainty_model, norm_bound):

    # Define graph object
    M = prMDP()
    M.set_initial_states(int(model.initial_states[0]))
    
    for s in model.states:
        
        print('- Add state {}'.format(s.id))
        M.states_dict[s.id] = state(s.id)
        
        # Check if this state is an initial state
        if s.id in model.initial_states:
            print('-- State {} is an initial state'.format(s.id))
            M.states_dict[s.id].initial = True
            
        # Set state as terminal by default
        M.states_dict[s.id].terminal = True
            
        for a in s.actions:
            print('-- Add action {} for state {} with {} transitions'.format(a.id, s.id, len(a.transitions)))
            M.states_dict[s.id].actions_dict[a.id] = action(a.id)
        
            successors      = np.array([t.column for t in a.transitions])
            probabilities   = np.array([t.value() for t in a.transitions])
            
            M.states_dict[s.id].actions_dict[a.id].successors = successors
            
            # Check if this is a self-loop
            if all(successors == s.id):
                # Self-loop
                M.states_dict[s.id].actions_dict[a.id].selfloop = True
                M.states_dict[s.id].actions_dict[a.id].deterministic = True
                
            elif len(successors) == 1:
                # State is nonterminal
                M.states_dict[s.id].terminal = False
                
                # Deterministic transition (no uncertainty model)
                M.states_dict[s.id].actions_dict[a.id].deterministic = True
                
                M.states_dict[s.id].actions_dict[a.id].model = \
                        distribution(successors, probabilities)
                        
            else:
                # State is nonterminal
                M.states_dict[s.id].terminal = False
                
                M.states_dict[s.id].actions_dict[a.id].robust = True
                
                # Determine margin between precise distribution and simplex
                norm_concat = np.concatenate([probabilities, 1-probabilities])
                norm_margin = np.min(np.abs(norm_concat))

                if norm_bound > norm_margin:
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