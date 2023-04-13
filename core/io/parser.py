import argparse
from ast import literal_eval

def parse_main(manualModel=None):
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
    
    ### INPUT MODEL ###    
    # Path to PRISM model to load
    parser.add_argument('--model', type=str, action="store", dest='model', 
                        default=manualModel, help="Path to PRISM model")
    
    # Path to PRISM model to load
    parser.add_argument('--instance', type=str, action="store", dest='instance', 
                        default=False, help="Name of the benchmark instance (used in exports)")
    
    # Path to parameter valuation file to load
    parser.add_argument('--parameters', type=str, action="store", dest='parameters', 
                        default=False, help="Path to parameter valuation file")
    
    parser.add_argument('--no_par_dependencies', dest='no_par_dependencies', action='store_true',
                        help="Avoid prMCs with parameters that are shared between states")
    parser.set_defaults(no_par_dependencies=False)
    
    # Path to parameter valuation file to load
    parser.add_argument('--true_param_file', type=str, action="store", dest='true_param_file', 
                        default=False, help="Path to True parameter valuation file (used for learning experiments)")
    
    # Temporal logic formula
    parser.add_argument('--formula', type=str, action="store", dest='formula', 
                        default=None, help="Formula to verify")
    
    parser.add_argument('--goal_label', type=str, action="store", dest='goal_label', 
                        default=None, help="For reachability probabilities, the label of the goal states (currently only support single label)")
    
    # Type of uncertainty model fo use
    parser.add_argument('--uncertainty_model', type=str, action="store", dest='uncertainty_model', 
                        default='Linf', help="Type of uncertainty model (Linf, L1, ...)")
    
    # Discount factor for expected rewards
    parser.add_argument('--discount', type=float, action="store", dest='discount', 
                        default=1, help="Discount factor")
    
    parser.add_argument('--beta_penalty', type=float, action="store", dest='beta_penalty', 
                        default=0, help="Penalty on dual variable beta as a tie-break rule")
    
    parser.add_argument('--default_valuation', type=float, action="store", dest='default_valuation', 
                        default=0.5, help="Default parameter valuation")
    
    parser.add_argument('--scale_reward', dest='scale_reward', action='store_true',
                        help="If True, rewards for prMC are normalized to one.")
    parser.set_defaults(scale_reward=False)
    
    
    
    ### PROGRAM SETTINGS ###  
    # CVX solver
    parser.add_argument('--solver', type=str, action="store", dest='solver', 
                        default='GUROBI', help="Solver to solve convex optimization problems with")
    
    parser.add_argument('--num_deriv', type=int, action="store", dest='num_deriv', 
                        default=1, help="Number of K derivatives to return")
    
    # Switch to enable verbose mode
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Verbose mode")
    parser.set_defaults(verbose=False)
    
    # Output folder
    parser.add_argument('--output_folder', type=str, action="store", dest='output_folder', 
                        default='output/', help="Folder in which to store results")
    
    parser.add_argument('--no_export', dest='no_export', action='store_true',
                        help="Avoid exporting results")
    parser.set_defaults(no_export=False)
    
    
    
    ### GRADIENT SETTINGS ###    
    # Switch to validate gradients empirically
    parser.add_argument('--validate_gradients', dest='validate_gradients', action='store_true',
                        help="Validate gradients by an empirical perturbation analysis")
    parser.set_defaults(validate_gradients=False)
    
    parser.add_argument('--validate_delta', type=float, action="store", dest='validate_delta', 
                        default=1e-4, help="Perturbation value to validate gradients")
    
    parser.add_argument('--explicit_baseline', dest='explicit_baseline', action='store_true',
                        help="Perform baseline, which computes all derivatives explicitly")
    parser.set_defaults(explicit_baseline=False)
    
    parser.add_argument('--no_gradient_validation', dest='no_gradient_validation', action='store_true',
                        help="If provided, validation of computed derivatives (gradient) is skipped")
    parser.set_defaults(no_gradient_validation=False)
    
    parser.add_argument('--robust_bound', type=str, action="store", dest='robust_bound', 
                        default='upper', help="Either 'upper' or 'lower' robust bound")    
    
    parser.add_argument('--robust_confidence', type=float, action="store", dest='robust_confidence', 
                        default=0.9, help="Confidence level on individual PAC probability intervals")
    
    
    
    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()    
    
    if args.goal_label is not None:
        args.goal_label = literal_eval(args.goal_label)
        assert type(args.goal_label) == set
    
    assert args.uncertainty_model in ['Linf', 'L1', 'hoeffding']
    
    assert args.robust_bound in ['upper', 'lower']
    
    return args