import argparse
from ast import literal_eval

def parse_inputs(manualModel=None):
    """
    Function to parse arguments provided

    Parameters
    ----------
    :manualModel: Override model as provided as argument in the command
    :nobisim: Override bisimulatoin option as provided as argument in the command

    Returns
    -------
    :args: Dictionary with all arguments

    """
    
    parser = argparse.ArgumentParser(description="Program to compute gradients for prMDPs")
    
    # Path to PRISM model to load
    parser.add_argument('--model', type=str, action="store", dest='model', 
                        default=manualModel, help="Path to PRISM model")
    
    # Temporal logic formula
    parser.add_argument('--formula', type=str, action="store", dest='formula', 
                        default=None, help="Formula to verify")
    
    # Set labels for terminal states
    parser.add_argument('--terminal_label', type=str, action="store", dest='terminal_label', 
                        default=('NONE'), help="Label for terminal states")
    
    # CVX solver
    parser.add_argument('--solver', type=str, action="store", dest='solver', 
                        default='GUROBI', help="Solver to solve convex optimization problems with")
    
    # Type of uncertainty model fo use
    parser.add_argument('--uncertainty_model', type=str, action="store", dest='uncertainty_model', 
                        default='L0', help="Type of uncertainty model (L0, L1, ...)")
    parser.add_argument('--uncertainty_size', type=float, action="store", dest='uncertainty_size', 
                        default=0.05, help="Size of the uncertainty set")
    
    # Switch to validate gradients empirically
    parser.add_argument('--validate_gradients', dest='validate_gradients', action='store_true',
                        help="Validate gradients by an empirical perturbation analysis")
    parser.set_defaults(validate_gradients=False)
    
    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()    

    if args.model == None:
        print('ERROR: No model specified')
        assert False
        
    # Interprete some arguments as lists
    if args.terminal_label:
        try:
            args.terminal_label = tuple(literal_eval(args.terminal_label))
            print('sucess')
        except:
            args.terminal_label = tuple([str(args.terminal_label)])
    else:
        tuple('<>')
            
    assert type(args.terminal_label) is tuple
    
    assert args.uncertainty_model in ['L0', 'L1']
    
    return args