# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:44:50 2022

@author: Thom Badings
"""

import numpy as np
import cvxpy as cp
import json
from tabulate import tabulate

import stormpy
import stormpy.core
import stormpy.examples
import stormpy.examples.files
import stormpy._config as config

from models.uncertainty_models import L0_polytope, L1_polytope

def parse_pmdp(path, formula, args, param_path = False, policy = 'optimal', verbose = False):

    print('Load PRISM model with STORM...')
    
    # Check support for parameters
    if not config.storm_with_pars:
        print("Support parameters is missing. Try building storm-pars.")
        return
    
    import stormpy.pars
    
    try:
        program = stormpy.parse_prism_program(path)
        properties = stormpy.parse_properties(formula, program)
        model = stormpy.build_parametric_model(program, properties)
    except:
        properties = stormpy.parse_properties(formula)
        model = stormpy.build_parametric_model_from_drn(path)
    
    terminal_states = []
    
    for state in model.states:
        successors = set()
        for action in state.actions:
            for transition in action.transitions:
                successors.add(transition.column)
                
        if successors == set({state.id}):
            terminal_states += [state.id]
    
    print("- Model supports parameters: {}".format(model.supports_parameters))
    parameters = model.collect_probability_parameters()
    print("- Number of parameters: {}".format(len(parameters)))
    
    instantiator = stormpy.pars.PDtmcInstantiator(model)
    point = dict()
    
    params2states = {}
    
    # Load parameter valuation
    
    if param_path:
        with open(param_path) as json_file:
            valuation = json.load(json_file)
    
    default_valuation = 0.5
    
    # Instantiate parameters
    for x in parameters:

        if param_path and x.name in valuation:
            point[x] = stormpy.RationalRF(float(valuation[x.name]))
        else:
            point[x] = stormpy.RationalRF(default_valuation)
            
        params2states[x] = set({})
        
    instantiated_model = instantiator.instantiate(point)
    
    for state in model.states:
        for action in state.actions:
            for transition in action.transitions:
                
                params = transition.value().gather_variables()
                
                for x in params:
                    params2states[x].add(state)
            
    
    # Compute model checking result
    result = stormpy.model_checking(instantiated_model, properties[0])
    print(result)
    
    return model, terminal_states, instantiated_model, np.array(list(parameters)), point, result, params2states

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