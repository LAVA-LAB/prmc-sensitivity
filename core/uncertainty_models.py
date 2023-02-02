# -*- coding: utf-8 -*-

import numpy as np
import itertools
from core.polynomial import polynomial

def Linf_polytope(center, size):
    
    '''
    Generate matrix inequalities for the Linf norm around 'center', bounded by
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
        b[2*d]   = polynomial(param = size, coeff = [-center[d], 1], power = [0, 1])
        b[2*d+1] = polynomial(param = size, coeff = [ center[d], 1], power = [0, 1])
        
    return A,b

def Hoeffding_interval(center, confidence, parameter, min_probability):
    
    '''
    Compute polytopic uncertainty set, representing intervals obtained using
    Hoeffding's inequality
    Returns polytope of the form Ap <= b
    '''
    
    alpha = np.sqrt(np.log(2/(1-confidence)) / 2)
    
    # Retrieve dimension of polytope
    n = len(center)

    # Initialize matrix
    A = np.kron(np.eye(n), np.array([[-1],[1]]))
    
    b = np.zeros((2*n), dtype=object)
    
    # Possibly add extra constraints to avoid probabilities outside [0,1]
    A_add = []
    b_add = []
    
    epsilon = alpha * np.sqrt(1/parameter.value)
    
    for d in range(n):
        # print('Interval for {} is [{}, {}]'.format(parameter, center[d]-epsilon, center[d]+epsilon))
        
        b[2*d]   = polynomial(param = parameter, coeff = [-center[d], alpha], power = [0, -0.5])
        b[2*d+1] = polynomial(param = parameter, coeff = [ center[d], alpha], power = [0, -0.5])
        
        # Check if probability interval goes beyond [0,1] interval
        if center[d] + epsilon > 1:
            # print('- Add constraint for {}, d={}, to keep upper bound below 1'.format(parameter, d))
            
            A_entry = np.zeros(n)
            A_entry[d] = 1
            
            A_add += [A_entry]
            b_add += [1]
            
        if center[d] - epsilon < min_probability:
            # print('- Add constraint for {}, d={}, to keep lower bound above {}'.format(parameter, d, min_probability))
            
            A_entry = np.zeros(n)
            A_entry[d] = -1
            
            A_add += [A_entry]
            b_add += [-min_probability]
            
    # If needed, add extra constraints to the polytope
    if len(A_add) > 0:
        A_add = np.array(A_add)
        b_add = np.array(b_add)
        
        A = np.concatenate((A, A_add))
        b = np.concatenate((b, b_add))
        
    return A,b

def L1_polytope(center, size):
    '''
    Generate matrix inequalities for the L1 norm around 'center', bounded by
    the given 'size'
    Returns polytope of the form Ap <= b

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
        b[i] = polynomial(param = size, coeff = [v @ center, 1], power = [0, 1])
            
    return A,b