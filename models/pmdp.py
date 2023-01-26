# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:44:50 2022

@author: Thom Badings
"""

from core.commons import tocDiff

import numpy as np
import json
import sys

import stormpy
import stormpy.core
import stormpy._config as config

def get_pmdp_reward_vector(model, point):
    
    reward_model = next(iter(model.reward_models.values()))
    R = np.zeros(len(model.states))
    
    for s in model.states:
        if not model.is_sink_state(s):
            if reward_model.has_state_rewards:
                R[s.id] = float(reward_model.get_state_reward(s.id).evaluate(point))
            elif reward_model.has_state_action_rewards:
                R[s.id] = float(reward_model.get_state_action_reward(s.id).evaluate(point))
            else:
                sys.exit()
                
    return R

def instantiate_pmdp(model, properties, parameters, valuation):
    
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
    
    return instantiated_model, point, params2states

def verify_pmdp(instantiated_model, properties):
    
    # Compute model checking result
    result = stormpy.model_checking(instantiated_model, properties[0])

    return result

def instantiate_verify_pmdp_exact(model, properties, parameters, valuation): 

    inst_checker = stormpy.pars.PDtmcExactInstantiationChecker(model)
    inst_checker.specify_formula(stormpy.ParametricCheckTask(properties[0].raw_formula, True))
    inst_checker.set_graph_preserving(True)
    env = stormpy.Environment()
    
    point = dict()
    
    params2states = {}
    
    for x in parameters:

        point[x] = stormpy.RationalRF(float(valuation[x.name]))
            
        params2states[x] = set({})
    
    result = inst_checker.check(env, point)
    
    return result, point, params2states

def parse_pmdp(path, formula, args, param_path = False, policy = 'optimal', verbose = False, exact = False):

    print('Load PRISM model with STORM...')
    
    # Check support for parameters
    if not config.storm_with_pars:
        print("Support parameters is missing. Try building storm-pars.")
        return
    
    import stormpy.pars
    
    if verbose:
        print('- Parse model...')
        
    try:
        program = stormpy.parse_prism_program(path)
        properties = stormpy.parse_properties(formula, program)
        model = stormpy.build_parametric_model(program, properties)
    except:
        properties = stormpy.parse_properties(formula)
        model = stormpy.build_parametric_model_from_drn(path)
    
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
    
    if verbose:
        print("- Instantiate and check model...")
    
    instantiated_model, point, params2states = instantiate_pmdp(model, properties, parameters, valuation)
    
    print("- Iterate once over all transitions...")
    
    # Store which parameters are related to which states
    for i,state in enumerate(model.states):
        if i % 10000 == 0 and verbose:
            print('-- Check state {}'.format(i))
        
        for action in state.actions:
            for transition in action.transitions:
                
                params = transition.value().gather_variables()
                
                for x in params:
                    params2states[x].add(state)
    
    return model, properties, np.array(list(parameters)), valuation, point, instantiated_model, params2states