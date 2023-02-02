import numpy as np
import os
from pathlib import Path
from datetime import datetime
from core.classes import PMC, PRMC
import sys

def export_json(args, MODEL, T, inst, solution, deriv, parameters = None):

    # Export results
    print('\nExport results...')

    import json
    
    # Check if a pMC or a prMC is provided
    if isinstance(MODEL, PMC):
        nr_states = MODEL.model.nr_states
        nr_transitions = MODEL.model.nr_transitions
        parameters = MODEL.parameters
        model_type = 'pMC'
        
    elif isinstance(MODEL, PRMC):
        nr_states = len(MODEL.states)
        nr_transitions = MODEL.nr_transitions
        model_type = 'prMC'
            
    else:
        sys.exit('ERROR: Unknown model type provided to export function')
        
    
    OUT = {
           'instance': args.model if not args.instance else args.instance,
           'Type': model_type,
           'Formula': args.formula,
           'Engine': args.pMC_engine,
           'States': nr_states,
           'Transitions': nr_transitions,
           'Parameters': len(inst['valuation']),
           #
           'Solution': np.round(solution, 6),
           'Model parse [s]': np.round(T['parse_model'], 6),
           'Model instantiate [s]': np.round(T['instantiate'], 6),
           'Model verify [s]': np.round(T['verify'], 6),
           #
           'Num. derivatives': args.num_deriv
           }
    
    if args.explicit_baseline:
        OUT['Differentiate one [s]'] = np.round(T['solve_explicit_one'], 6)
        OUT['Differentiate explicitly [s]'] = np.round(T['solve_explicit_all'], 3)
    
    OUT['LP (define matrices) [s]'] = np.round(T['build_matrices'], 6)
    OUT['LP (solve) [s]'] = np.round(T['solve_LP'], 6)
    
    if args.num_deriv > 1:
        OUT['Max. derivatives'] = list(np.round(deriv['LP'], 6))
        OUT['Argmax. derivatives'] = [p.name for p in parameters[ deriv['LP_idxs']]]
        
        if not args.no_gradient_validation:
            OUT['Max. validation'] = list(np.round(deriv['validate'], 6))
            OUT['Difference %'] = list(np.round(deriv['RelDiff'], 6))
        
    else:
        OUT['Max. derivatives'] = np.round(deriv['LP'][0], 6)
        OUT['Argmax. derivatives'] = deriv['LP_pars']
        
        if not args.no_gradient_validation:
            OUT['Max. validation'] = np.round(deriv['validate'][0], 6)
            OUT['Difference %'] = np.round(deriv['RelDiff'][0], 6)
    
    output_path = Path(args.root_dir, args.output_folder)
    if not os.path.exists(output_path):
    
       # Create a new directory because it does not exist
       os.makedirs(output_path)
       
    dt = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    out_file = os.path.join(args.output_folder, Path(args.model).stem + "_" + model_type + dt + '.json')
    with open(str(Path(args.root_dir, out_file)), "w") as outfile:
        json.dump(OUT, outfile)