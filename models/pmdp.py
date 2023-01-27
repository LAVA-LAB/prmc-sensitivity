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
    
    params2states = get_parameters_to_states(model, params2states)
    
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
    
    params2states = get_parameters_to_states(model, params2states)
    
    return result, point, params2states

def parse_pmdp(path, args, param_path = False, policy = 'optimal', verbose = False):

    print('Load PRISM model with STORM...')
    
    # Check support for parameters
    if not config.storm_with_pars:
        print("Support parameters is missing. Try building storm-pars.")
        return
    
    import stormpy.pars
    
    if verbose:
        print('- Parse model...')
        
    if path.suffix == '.drn':
        properties = stormpy.parse_properties(args.formula)
        model = stormpy.build_parametric_model_from_drn(str(path))
    else:        
        program = stormpy.parse_prism_program(str(path))
        properties = stormpy.parse_properties(args.formula, program)
        model = stormpy.build_parametric_model(program, properties)
        
    
    parameters = model.collect_probability_parameters()
    print("- Number of parameters: {}".format(len(parameters)))
    
    # Load parameter valuation
    if param_path:
        with open(str(param_path)) as json_file:
            valuation_raw = json.load(json_file)
            valuation = {}
            sample_size = {}
            
            for v,val in valuation_raw.items():
                if type(val) == list:
                    valuation[v],sample_size[v] = val
                    
                else:
                    valuation = valuation_raw
                    sample_size = None
                    break
            
    else:
        valuation = {}
        sample_size = None
        
        for x in parameters:
            valuation[x.name] = args.default_valuation
    
    if len(model.reward_models) == 0 and args.pMC_engine == 'spsolve':
        print('\nWARNING: verifying using spsolve requires a reward model, but none is given.')
        print('>> Switch to Storm for verifying model.\n')
        args.pMC_engine = 'storm'
        
        # Storm often needs a larger perturbation delta to get reliable validation results
        mindelta = 1e-3
        
        if args.validate_delta < mindelta:
            print('>> Set parameter delta to {}'.format(mindelta))
            args.validate_delta = mindelta
    
    return model, properties, np.array(list(parameters)), valuation, sample_size

def get_parameters_to_states(model, params2states):
    
    print("- Obtain mapping from parameters to states...")
    
    # Store which parameters are related to which states
    for i,state in enumerate(model.states):
        for action in state.actions:
            for transition in action.transitions:
                
                params = transition.value().gather_variables()
                
                for x in params:
                    params2states[x].add(state)
                    
    return params2states