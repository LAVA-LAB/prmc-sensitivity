# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:20:35 2022

@author: Thom Badings
"""

import numpy as np
from poly import poly,  polytope
import cvxpy as cp

params = {
    'par1': cp.Parameter(value = 0.1),
    'par2': cp.Parameter(value = 0.2),
    'par3': cp.Parameter(value = 0.9),
    }

A = np.array([
    [ poly(params['par1'], {1: 2}), poly(params['par1'], {2: 0.5}) ],
    [ poly(params['par1'], {0: 3}), poly(params['par2'], {4: 2}) ],
    [ 1, 0]
    ])

b = [
    poly(params['par1'], [0, 2]),
    poly(params['par2'], [0, 1]),
    poly(params['par1'], [0, 1]),
    ]

PT = polytope(A, b)

# b2 = list(map(lambda x: x.deriv_eval(params['par1']), b))

apply = np.vectorize(lambda x,v: x.deriv_eval(v) if isinstance(x, poly) else 0, 
                     otypes=[float])

dA = apply(A, params['par1'])

db = apply(b, params['par1'])