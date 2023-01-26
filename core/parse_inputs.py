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
    
    parser.add_argument('--pMC_engine', type=str, action="store", dest='pMC_engine', 
                        default='spsolve', help="Engine to solve pMCs with")
    
    # parser.add_argument('--exact', dest='exact', action='store_true',
    #                     help="Perform exact model checking")
    # parser.set_defaults(exact=False)
    
    # Path to parameter valuation file to load
    parser.add_argument('--parameters', type=str, action="store", dest='parameters', 
                        default=False, help="Path to parameter valuation file")
    
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
    
    parser.add_argument('--validate_delta', type=float, action="store", dest='validate_delta', 
                        default=1e-3, help="Step ")
    
    # Switch to enable verbose mode
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Verbose mode")
    parser.set_defaults(verbose=False)
    
    # Discount factor for expected rewards
    parser.add_argument('--discount', type=float, action="store", dest='discount', 
                        default=1, help="Discount factor")
    
    parser.add_argument('--beta_penalty', type=float, action="store", dest='beta_penalty', 
                        default=1e-9, help="Penalty on dual variable beta as a tie-break rule")
    
    parser.add_argument('--default_valuation', type=float, action="store", dest='default_valuation', 
                        default=0.5, help="Default parameter valuation")
    
    # Output folder
    parser.add_argument('--output_folder', type=str, action="store", dest='output_folder', 
                        default='output/', help="Folder in which to store results")
    
    parser.add_argument('--no_export', dest='no_export', action='store_true',
                        help="Avoid exporting results")
    parser.set_defaults(no_export=False)
    
    
    
    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()    
        
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
    
    assert args.pMC_engine in ['storm', 'spsolve']
    
    return args