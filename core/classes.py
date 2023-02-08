import numpy as np
import json
import stormpy
import stormpy.core
import stormpy._config as config
from tabulate import tabulate

from core.uncertainty_models import Linf_polytope, L1_polytope, Hoeffding_interval

class PMC:
    
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



class PRMC:
    
    def __init__(self, num_states):
        
        self.states_dict = {}
        self.parameters = {}
        self.sample_size = {}
        self.parameters_max_value = {}
        
        self.stateAction2param = {}
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

    def set_reward_models(self, rewards):
        
        self.rewards = rewards
        
    def get_state_set(self):
        
        return set(self.states_dict.keys())
    
    def update_distribution(self, var, inst):
        '''
        Update a single parameter v of the PRMC
        '''
        
        for (s,a) in self.param2stateAction[ var ]:
            
            SA = self.states_dict[s].actions_dict[a]
            probabilities = np.array([float(t.value().evaluate(inst['point'])) for t in SA.parametricTrans])
            successors = SA.successors
            
            if len(successors) == 1:
                
                SA.model = distribution(successors, probabilities)
                
            else:
                
                # Update probability distribution
                SA.model.update_point(probabilities)
                
                

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
    
    def __init__(self, A, b, typ, parameter, confidence = None):
        
        self.A = A
        self.b = b
        self.parameter = parameter
        self.type = typ
        self.confidence = confidence
        
    def update_point(self, probabilities):
        
        if self.type == Hoeffding_interval:
            A, b = Hoeffding_interval(probabilities, self.confidence, self.parameter)
            
        else:
            A, b = self.type(probabilities, self.parameter)
            
        self.A = A
        self.b = b