# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:05:56 2022

@author: Thom Badings
"""

# %run "~/documents/sensitivity-prmdps/prmdp-sensitivity-git/run_ext.py"

from core.sensitivity import gradients_cvx
from core.commons import tocDiff
from core.cvx import verify_cvx
from models.load_model import load_prism_model
from models.parse_model import parse_storm
from models.uncertainty_models import L0_polytope, L1_polytope

import numpy as np
import os

from core.commons import unit_vector, valuate, deriv_valuate
from core.parse_inputs import parse_inputs

# Set root directory
root_dir = os.path.dirname(os.path.abspath(__file__))

# Parse arguments
args = parse_inputs()

print('---------------------------------------')
print('START PROGRAM WITH ARGUMENTS:')
print('Path to model:     ', args.model)
print('Formula:           ', args.formula)
print('Term. state label: ', args.terminal_label)
print('Uncertainty model: ', args.uncertainty_model)
print('Size of unc. sets: ', args.uncertainty_size)
print('---------------------------------------\n')

# Load PRISM model with STORM
model, policy = load_prism_model(args.model, args.formula)
policy = None

if args.uncertainty_model == 'L0':
    uncertainty_model = L0_polytope
else:
    uncertainty_model = L1_polytope
L1_size = args.uncertainty_size

# Parse model and policy
M = parse_storm(model, policy, uncertainty_model, L1_size, args.terminal_label)

# Hyperparameters for optimization problem
M.DISCOUNT          = 1.00
M.PENALTY           = 1e-6
M.DELTA     	    = 1e-6

# %%

# Verify loaded model by solving the optimization program
CVX = verify_cvx(M, verbose=True)
CVX.solve(solver = args.solver, store_initial = True, verbose = True)

print('\nValue of the measure in initial state:', CVX.prob.value)

# Check if complementary slackness is satisfied
CVX.check_complementary_slackness()

# %%    

'''
SVT = gradients_cvx(M, CVX.x, CVX.alpha, CVX.beta, CVX.cns)

tocDiff(False)

GRAD_old = np.zeros(len(M.parameters))

for i, (THETA_SA,THETA) in enumerate(M.parameters.items()):

    Vx = SVT.solve(M, THETA, args.solver)
    print('\nParameter {} solved in {} seconds'.format(THETA_SA, tocDiff(False)))
    print('Gradient in initial state:', Vx[list(M.sI.keys())].value)
    
    grad_analytical   = Vx.value[0:len(M.states)]
    cum_diff = CVX.delta_solve(THETA, M.DELTA, grad_analytical, args.solver, verbose = False)
    
    print('- Cum.diff. between analytical and numerical is {:.3f}'.format(cum_diff))
    
    GRAD_old[i] = Vx[list(M.sI.keys())].value
'''

# %%

tocDiff(False)


from core.sensitivity import gradients_spsolve

GRAD = gradients_spsolve(M, CVX)[0,:]
print('\nLinear equation system for {} parameters solved in {} seconds'.format(len(M.parameters), tocDiff(False)))