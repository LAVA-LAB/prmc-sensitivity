import numpy as np
import scipy.sparse as sparse
import time
import random

def explicit_gradient(pmc, args, J, Ju):
        
    start_time = time.time()
    
    # Select N random parameters
    idxs = np.arange(len(pmc.parameters))
    random.shuffle(idxs)
    sample_idxs = idxs[:min(len(pmc.parameters), 100)]
    
    deriv_expl = np.zeros(len(sample_idxs), dtype=float)
    
    for i,(q,x) in enumerate(zip(sample_idxs, pmc.parameters[sample_idxs])):
        
        deriv_expl[i] = sparse.linalg.spsolve(J, -Ju[:,q])[pmc.sI['s']] @ pmc.sI['p']
        
    time_solve_explicit_one = (time.time() - start_time) / len(sample_idxs)
    time_solve_explicit_all = time_solve_explicit_one * len(pmc.parameters)
    
    return deriv_expl, time_solve_explicit_one, time_solve_explicit_all