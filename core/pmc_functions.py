from core.sensitivity import solve_cvx_gurobi
from core.baseline_gradient import explicit_gradient
from core.verify_pmc import verify_spsolve, verify_pmdp_storm, verify_pmc_storm, \
    define_sparse_LHS, verify_linear_program

import numpy as np
import scipy.sparse as sparse
import time
import stormpy
import json
import sys

    

def pmc_derivative_LP(pmc, J, Ju, args, T = False):    
    '''
    Compute the k=args.num_deriv highest derivatives for a pMC.

    Parameters
    ----------
    pmc : pMC object
    J : Left-hand side sparse matrix (CSR format)
    Ju : Right-hand side matrix (CSC format)
    args : Arguments object
    T : Timing object, used to store run times

    Returns
    -------
    deriv : Dictionary with derivative results

    '''    
    
    assert args.num_deriv <= Ju.shape[1]
    
    # Upper bound number of derivatives to the number of parameters
    args.num_deriv = min(args.num_deriv, len(pmc.parameters))

    ### Compute K most important parameters

    deriv = {}

    print('Compute parameter importance via LP (GurobiPy)...')
    start_time = time.time()

    optm, deriv['LP_idxs'], deriv['LP'] = solve_cvx_gurobi(J, Ju, pmc.sI, args.num_deriv,
                                        direction=args.derivative_direction, verbose=True)

    deriv['LP_pars'] = pmc.parameters[ deriv['LP_idxs']][0].name

    if T:
        T.times['derivative_LP'] = time.time() - start_time   
        print('- LP solved in: {:.3f} sec.'.format(T.times['derivative_LP']))
    print('- Obtained derivatives are {} for parameters {}'.format(deriv['LP'],  deriv['LP_pars']))

    if args.explicit_baseline:
        print('-- Execute baseline: compute all gradients explicitly')
        
        deriv['explicit'] = explicit_gradient(pmc, args, J, Ju, T)
            
    return optm, deriv


def pmc_validate_derivative(pmc, inst, solution, deriv, delta):
    '''
    Validate derivatives numerically, by giving each parameter a small delta
    and recomputing the solution

    Parameters
    ----------
    pmc : pMC object
    inst : Parameter instantiation dictionary
    solution : Current solution (scalar value)
    deriv : Dictionary with derivative results
    delta : delta to give to the parameter instantiation

    Returns
    -------
    deriv : Updated derivatives object

    '''

    print('\nValidation by perturbing parameters by +{}'.format(delta))
    
    deriv['validate'] = np.zeros(len(deriv['LP_idxs']), dtype=float)
    deriv['RelDiff']  = np.zeros(len(deriv['LP_idxs']), dtype=float)
    
    for q,x in enumerate(pmc.parameters[deriv['LP_idxs']]):
        
        # Increment this parameter by the given delta
        inst['valuation'][x.name] += delta
        
        # instantiate model
        instantiated_model, point = pmc_instantiate(pmc, inst['valuation'])
        
        # After that, use spsolve to obtain the actual solution    
        _, result = verify_linear_program(instantiated_model, pmc.reward, pmc.scheduler_prob)
        # _, result = verify_spsolve(instantiated_model, pmc.reward, pmc.scheduler_prob)
            
        # Extract solution
        solution_new = result[pmc.sI['s']] @ pmc.sI['p']
        
        # Compute derivative
        deriv['validate'][q] = (solution_new-solution) / delta
        
        # Determine difference in %
        if deriv['LP'][q] != 0:
            deriv['RelDiff'][q] = (deriv['validate'][q]-deriv['LP'][q])/deriv['LP'][q]
        
        print('- Parameter {}, LP: {:.6f}, val: {:.6f}, diff: {:.6f}'.format(x, deriv['LP'][q], deriv['validate'][q], deriv['RelDiff'][q]))
        
        inst['valuation'][x.name] -= delta
            
    return deriv


def pmc_load_instantiation(pmc, param_path, default_valuation):
    '''
    Load parameter instantiation for a pMC

    Parameters
    ----------
    pmc : pMC object
    param_path : Path (string) to instantiation file to load.
    default valuation : Scalar value for the default valuation of a parameter

    Returns
    -------
    inst : Instantiation dictionary

    '''
    
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
        
        for x in pmc.parameters:
            valuation[x.name] = default_valuation
            
    inst = {'valuation': valuation,
            'sample_size': sample_size}
            
    return inst


def pmc_instantiate(pmc, valuation, T = False):
    '''
    Instantiate a pMC with the given valuation

    Parameters
    ----------
    pmc : pMC object
    valuation : Parameter valuation (dictionary)
    T : Timing object, used to store run times

    Returns
    -------
    instantiated_model : Stormpy instantiated model
    point : Parameter point

    '''
    
    start_time = time.time()
    
    if pmc.model.model_type.name == 'MDP':
        instantiator = stormpy.pars.PMdpInstantiator(pmc.model)
    else:
        instantiator = stormpy.pars.PDtmcInstantiator(pmc.model)
        
    point = dict()
    
    for x in pmc.parameters:
        point[x] = stormpy.RationalRF(float(valuation[x.name]))
        
    instantiated_model = instantiator.instantiate(point)
    
    if T:
        T.times['instantiate'] = time.time() - start_time
        
    return instantiated_model, point

def assert_probabilities(model):
    '''
    Assert if all probability distributions of given model are valid.

    '''
    
    for s in model.states:
        for a in s.actions:
            distr = np.array([t.value() for t in a.transitions])
            assert np.all(distr >= 0) and np.all(distr <= 1)
            assert np.isclose(np.sum(distr), 1)