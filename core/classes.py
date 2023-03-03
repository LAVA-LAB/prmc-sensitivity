import numpy as np
import stormpy._config as config

from tabulate import tabulate
from core.uncertainty_models import Linf_polytope, L1_polytope, Hoeffding_interval

class PMC:
    
    def __init__(self, model_path, args, verbose = False):
        
        self.model_path = model_path
        self.verbose = verbose
        
        # Load PRISM model
        self.model, self.properties, self.parameters = self._load_prism_model(args)
        
        # Define initial state
        self.sI = {'s': np.array(self.model.initial_states), 
                   'p': np.full(len(self.model.initial_states), 1/len(self.model.initial_states))}
        
        
    def _load_prism_model(self, args):
        
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
        
        self.param2stateAction = {}
        
        self.robust_successors = {}
        self.robust_constraints = 0
        
    def __str__(self):
        
        items = {
            'No. states': len(self.states),
            'No. parameters': len(self.parameters),
            'Robust constraints': self.robust_constraints,
            'Discount factor': self.discount
            }
        
        print_list = [[k,v] for (k,v) in items.items()]
        
        return '\n' + tabulate(print_list, headers=["Property", "Value"]) + '\n'
        
    def set_state_iterator(self):
        
        self.states = list(self.states_dict.values())
        
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
            
            # print('New point estimate for {} is: {}'.format(var.name, probabilities))
            # print('b before:', SA.model.b)
            
            if len(successors) == 1:
                
                SA.model = distribution(successors, probabilities)
                
            else:
                
                # Update probability distribution
                SA.model.update_point(probabilities)
                
            # print('b after:', SA.model.b)
                
    def set_robust_constraints(self):
        
        self.robust_constraints = 0
        
        for s in self.states:
            for a in s.actions:
                if a.robust:
                
                    # Put an (arbitrary) ordering over the dual variables
                    a.alpha_start_idx = self.robust_constraints
                    self.robust_constraints += len(a.model.b)
                    
        return

                

class state:
    
    def __init__(self, id):
        
        self.id = id
        self.initial = False
        self.actions_dict = {}

    def set_action_iterator(self):
        
        self.actions = list(self.actions_dict.values())
        
        
        
class action:
    
    def __init__(self, id):
        
        self.id = id
        self.model = None # Uncertainty model
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
