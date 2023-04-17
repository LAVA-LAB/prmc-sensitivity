from core.sensitivity import solve_cvx_single

import numpy as np
import scipy.sparse as sparse
import time
import random

def explicit_gradient(model, args, J, Ju, T = False, N = 10):
    '''
    Compute N derivatives explicitly by solving the respecitve equation system
    of J x = Ju, with x the vector of derivatives.

    Parameters
    ----------
    model : PMC or PRMC object
    args : Arguments object
    J : Left-hand side sparse matrix (CSR format)
    Ju : Right-hand side matrix (CSC format)
    T : Timing object, used to store run times

    Returns
    -------
    deriv_expl : Vector of derivatives

    '''    
    
    # Select N random parameters
    idxs = np.arange(len(model.parameters))
    random.shuffle(idxs)
    sample_idxs = idxs[:min(len(model.parameters), N)]
    
    deriv_expl = np.zeros(len(sample_idxs), dtype=float)
    
    print('\nStart baseline of computing all derivatives explicitly')

    solve_time_sum = 0

    for i,q in enumerate(sample_idxs):
        print('--- Compute derivative nr. {}'.format(i))
        
        # If matrix is square, use sparse matrix solver
        if J.shape[0] == J.shape[1]:
            
            start_time = time.time()
            deriv_expl[i] = sparse.linalg.spsolve(J, -Ju[:,q])[model.sI['s']] @ model.sI['p']
            solve_time_sum += time.time() - start_time 

        # Otherwise, solve via LP
        else:
            print('> Perform baseline (computing all derivatives) using LP')

            deriv_expl[i], solve_time = solve_cvx_single(J, Ju[:,q], model.sI, method = 2)
            solve_time_sum += solve_time

            T.notes += ['Baseline for gradients with LP']
        
    if T:
        T.times['solve_one_derivative'] = solve_time_sum / len(sample_idxs)
        T.times['solve_all_derivatives'] = T.times['solve_one_derivative'] * len(model.parameters)
    
    return deriv_expl
