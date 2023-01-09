# -*- coding: utf-8 -*-

import numpy as np
import itertools
from core.poly import poly

def L0_polytope(center, size):
    
    '''
    Generate matrix inequalities for the L0 norm around 'center', bounded by
    the given 'size', such that Ap <= b and

    Returns
    -------
    A matrix and b vector

    '''
    
    # Retrieve dimension of polytope
    n = len(center)

    # Initialize matrix
    A = np.kron(np.eye(n), np.array([[-1],[1]]))
    
    b = np.zeros((2*n), dtype=object)
    
    for d in range(n):
        b[2*d]   = poly(param = size, coeff = {0: -np.round(center[d], 3), 1: 1})
        b[2*d+1] = poly(param = size, coeff = {0: np.round(center[d], 3), 1: 1})
            
    return A,b

def L1_polytope(center, size):
    '''
    Generate matrix inequalities for the L1 norm around 'center', bounded by
    the given 'size'

    Returns
    -------
    A matrix and b vector

    '''
    
    # Retrieve dimension of polytope
    n = len(center)

    # Initialize matrix
    A = np.zeros((2**n, n), dtype=float)
    b = np.zeros((2**n), dtype=object)
    
    # Iterate over all possible perpendicular vectors
    for i,v in enumerate(itertools.product(*[[-1, 1]]*n)):
        
        v = np.array(v)
        
        # Row in a matrix is equal to vector v
        A[i] = v
        
        # Define b vector as a polytope object 
        b[i] = poly(param = size, coeff = {0: v @ center, 1: 1})
            
    return A,b