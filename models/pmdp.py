# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:44:50 2022

@author: Thom Badings
"""

from core.commons import tocDiff

import numpy as np
import json

import stormpy
import stormpy.core
import stormpy.examples
import stormpy.examples.files
import stormpy._config as config

def instantiate_pmdp(parameters, valuation, model):
    
    if model.model_type.name == 'MDP':
        instantiator = stormpy.pars.PMdpInstantiator(model)
    else:
        instantiator = stormpy.pars.PDtmcInstantiator(model)
        
    point = dict()
    
    params2states = {}
    
    for x in parameters:

        point[x] = stormpy.RationalRF(float(valuation[x.name]))
            
        params2states[x] = set({})
        
    instantiated_model = instantiator.instantiate(point)
    
    return instantiated_model, params2states, point

def parse_pmdp(path, formula, args, param_path = False, policy = 'optimal', verbose = False):

    print('Load PRISM model with STORM...')
    
    # Check support for parameters
    if not config.storm_with_pars:
        print("Support parameters is missing. Try building storm-pars.")
        return
    
    import stormpy.pars
    
    print('- Parse model...')
    tocDiff(False)
    
    try:
        program = stormpy.parse_prism_program(path)
        properties = stormpy.parse_properties(formula, program)
        model = stormpy.build_parametric_model(program, properties)
    except:
        properties = stormpy.parse_properties(formula)
        model = stormpy.build_parametric_model_from_drn(path)
    
    print('-- Model parsed in {} seconds'.format(tocDiff(False)))
    print('- Loop once over state space...')
    
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
    
    # Load parameter valuation
    if param_path:
        with open(param_path) as json_file:
            valuation = json.load(json_file)
    else:
        valuation = {}
        
        for x in parameters:
            valuation[x.name] = args.default_valuation
    
    print("- Instantiate model...")
    
    # Instantiate parameters
    instantiated_model, params2states, point = instantiate_pmdp(parameters, valuation, model)
    
    print("- Model checking...")
    
    # Compute model checking result
    result = stormpy.model_checking(instantiated_model, properties[0])
    
    print("- Iterate once over all transitions...")
    
    # Store which parameters are related to which states
    for i,state in enumerate(model.states):
        if i % 10000 == 0:
            print('-- Check state {}'.format(i))
        
        for action in state.actions:
            for transition in action.transitions:
                
                params = transition.value().gather_variables()
                
                '''
                # Gather variables related to this transition
                var = transition.value().gather_variables()
                
                # Get valuation for only those parameters
                subpoint = {v: point[v] for v in var}
                
                p = transition.value().evaluate(subpoint)
                if p > 1 or p < 0:
                    print('-- WARNING: transition probability for ({},{},{}) is {}'.format(state.id, action.id, transition.column, p))
                '''
                
                for x in params:
                    params2states[x].add(state)
    
    return model, properties, terminal_states, instantiated_model, np.array(list(parameters)), valuation, point, result, params2states