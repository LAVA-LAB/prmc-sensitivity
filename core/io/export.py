import numpy as np
import os
from pathlib import Path
from datetime import datetime
from core.classes import PMC, PRMC
import sys

class timer:
    def __init__(self):
        self.times = {}
        self.notes = []

def export_json(args, MODEL, T, inst, solution, deriv):

    # Export results
    print('\nExport results...')

    import json
    
    # Check if a pMC or a prMC is provided
    if isinstance(MODEL, PMC):
        nr_states = MODEL.model.nr_states
        nr_transitions = MODEL.model.nr_transitions
        nr_parameters = len(MODEL.parameters)
        model_type = 'pMC'
        
        parameters = MODEL.parameters
        output_pars = [p.name for p in parameters[ deriv['LP_idxs']]]
        
    elif isinstance(MODEL, PRMC):
        nr_states = len(MODEL.states)
        nr_transitions = MODEL.nr_transitions
        nr_parameters = len(MODEL.parameters)
        model_type = 'prMC'
        
        parameters = MODEL.parameters
        output_pars = [MODEL.parameters[sa].name() for sa in MODEL.paramIndex[deriv['LP_idxs']]]
        
    else:
        sys.exit('ERROR: Unknown model type provided to export function')
        
    
    OUT = {
           'Instance': args.model if not args.instance else args.instance,
           'Type': model_type,
           'Formula': args.formula,
           'States': nr_states,
           'Transitions': nr_transitions,
           'Parameters': nr_parameters,
           #
           'Solution': np.round(solution, 6),
           'Model instantiate [s]': np.round(T.times['initialize_model'], 6),
           'Model verify [s]': np.round(T.times['verify_model'], 6),
           #
           'Num. derivatives': args.num_deriv,
           'Notes': "; ".join(T.notes)
           }
    
    if args.explicit_baseline:
        OUT['Solve one derivative [s]'] = np.round(T.times['solve_one_derivative'], 6)
        OUT['Solve all derivatives [s]'] = np.round(T.times['solve_all_derivatives'], 3)
    
    OUT['Compute LHS matrices [s]'] = np.round(T.times['compute_LHS_matrix'], 6)
    OUT['Compute RHS matrices [s]'] = np.round(T.times['compute_RHS_matrix'], 6)
    OUT['Derivative LP (build and solve) [s]'] = np.round(T.times['solve_k_highest_derivatives'], 6)
    
    if args.num_deriv > 1:
        OUT['Max. derivatives'] = list(np.round(deriv['LP'], 6))
        OUT['Argmax. derivatives'] = output_pars
        
        if not args.no_gradient_validation:
            OUT['Max. validation'] = list(np.round(deriv['validate'], 6))
            OUT['Difference %'] = list(np.round(deriv['RelDiff'] * 0.01, 6))
        
    else:
        OUT['Max. derivatives'] = np.round(deriv['LP'][0], 6)
        OUT['Argmax. derivatives'] = output_pars
        
        if not args.no_gradient_validation:
            OUT['Max. validation'] = np.round(deriv['validate'][0], 6)
            OUT['Difference %'] = np.round(deriv['RelDiff'][0] * 0.01, 6)
    
    output_path = Path(args.root_dir, args.output_folder)
    if not os.path.exists(output_path):
    
       # Create a new directory because it does not exist
       os.makedirs(output_path)
       
    dt = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    out_file = os.path.join(args.output_folder, Path(args.model).stem + "_" + model_type + dt + '.json')
    with open(str(Path(args.root_dir, out_file)), "w") as outfile:
        json.dump(OUT, outfile)