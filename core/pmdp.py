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
import copy

class pMDP:
    
    def __init__(self, model_path, args,
                 policy = 'optimal', verbose = False):
        
        self.model_path = model_path
        self.policy = policy
        self.verbose = verbose
        
        # Load PRISM model
        self.model, self.properties, self.parameters = self.load_prism_model(args)
        
        if len(self.model.reward_models) == 0 and args.pMC_engine == 'spsolve':
            print('\nWARNING: verifying using spsolve requires a reward model, but none is given.')
            print('>> Switch to Storm for verifying model.\n')
            args.pMC_engine = 'storm'
            
            # Storm often needs a larger perturbation delta to get reliable validation results
            mindelta = 1e-3
            
            if args.validate_delta < mindelta:
                print('>> Set parameter delta to {}'.format(mindelta))
                args.validate_delta = mindelta
        
        # Define initial state
        self.sI = {'s': np.array(self.model.initial_states), 
                   'p': np.full(len(self.model.initial_states), 1/len(self.model.initial_states))}
        
        
    def load_prism_model(self, args):
        
        print('Load PRISM model with STORM...')
        
        # Check support for parameters
        if not config.storm_with_pars:
            print("Support parameters is missing. Try building storm-pars.")
            return
        
        import stormpy.pars
        
        if self.verbose:
            print('- Parse model...')
            
        if self.model_path.suffix == '.drn':
            properties = stormpy.parse_properties(args.formula)
            model = stormpy.build_parametric_model_from_drn(str(self.model_path))
        else:        
            program = stormpy.parse_prism_program(str(self.model_path))
            properties = stormpy.parse_properties(args.formula, program)
            model = stormpy.build_parametric_model(program, properties)
            
        parameters_set = model.collect_probability_parameters()
        parameters = np.array(list(parameters_set))
        print("- Number of parameters: {}".format(len(parameters)))
        
        return model, properties, parameters
        
        
    def load_instantiation(self, args, param_path):
        
        # Load parameter valuation
        if param_path:
            with open(str(param_path)) as json_file:
                valuation_raw = json.load(json_file)
                valuation = {}
                sample_size = {}
                
                for v,val in valuation_raw.items():
                    if type(val) == list:
                        valuation[v],sample_size[v] = val
                        
                        sample_size[v] = int(sample_size[v])
                        
                    else:
                        valuation = valuation_raw
                        sample_size = None
                        break
                
        else:
            valuation = {}
            sample_size = None
            
            for x in self.parameters:
                valuation[x.name] = args.default_valuation
                
        return valuation, sample_size
        
    
    def instantiate(self, valuation):
        
        if self.model.model_type.name == 'MDP':
            instantiator = stormpy.pars.PMdpInstantiator(self.model)
        else:
            instantiator = stormpy.pars.PDtmcInstantiator(self.model)
            
        point = dict()
        
        for x in self.parameters:
            point[x] = stormpy.RationalRF(float(valuation[x.name]))
            
        instantiated_model = instantiator.instantiate(point)
        
        return instantiated_model, point


    def get_parameters_to_states(self):
        
        print("- Obtain mapping from parameters to states...")
        
        params2states = {x: set({}) for x in self.parameters}
        
        # Store which parameters are related to which states
        for i,state in enumerate(self.model.states):
            for action in state.actions:
                for transition in action.transitions:
                    
                    params = transition.value().gather_variables()
                    
                    for x in params:
                        params2states[x].add(state)
                        
        return params2states


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



def verify_pmdp_storm(instantiated_model, properties, sI):
    
    # Compute model checking result
    result = stormpy.model_checking(instantiated_model, properties[0])
    array  = np.array(result.get_values(), dtype=float)

    return array

# def instantiate_verify_pmdp_exact(model, properties, parameters, valuation): 

#     inst_checker = stormpy.pars.PDtmcExactInstantiationChecker(model)
#     inst_checker.specify_formula(stormpy.ParametricCheckTask(properties[0].raw_formula, True))
#     inst_checker.set_graph_preserving(True)
#     env = stormpy.Environment()
    
#     point = dict()
    
#     params2states = {}
    
#     for x in parameters:

#         point[x] = stormpy.RationalRF(float(valuation[x.name]))
            
#         params2states[x] = set({})
    
#     result = inst_checker.check(env, point)
    
#     params2states = get_parameters_to_states(model, params2states)
    
#     return result, point, params2states



# def parse_pmdp(path, args, param_path = False, policy = 'optimal', verbose = False):

    
    
#     # Load parameter valuation
#     if param_path:
#         with open(str(param_path)) as json_file:
#             valuation_raw = json.load(json_file)
#             valuation = {}
#             sample_size = {}
            
#             for v,val in valuation_raw.items():
#                 if type(val) == list:
#                     valuation[v],sample_size[v] = val
                    
#                 else:
#                     valuation = valuation_raw
#                     sample_size = None
#                     break
            
#     else:
#         valuation = {}
#         sample_size = None
        
#         for x in parameters:
#             valuation[x.name] = args.default_valuation
    
#     if len(model.reward_models) == 0 and args.pMC_engine == 'spsolve':
#         print('\nWARNING: verifying using spsolve requires a reward model, but none is given.')
#         print('>> Switch to Storm for verifying model.\n')
#         args.pMC_engine = 'storm'
        
#         # Storm often needs a larger perturbation delta to get reliable validation results
#         mindelta = 1e-3
        
#         if args.validate_delta < mindelta:
#             print('>> Set parameter delta to {}'.format(mindelta))
#             args.validate_delta = mindelta
    
#     return model, properties, np.array(list(parameters)), valuation, sample_size