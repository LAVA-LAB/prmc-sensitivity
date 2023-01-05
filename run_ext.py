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
from models.parse_model import parse_storm, parse_policy, generate_random_policy
from models.uncertainty_models import L0_polytope, L1_polytope

SOLVER = 'GUROBI'

# model, policy = load_prism_model(stormpy.examples.files.prism_mdp_maze,
#                                   "Rmax=? [F \"goal\"]")
# TERM_LABELS = ('goal')

# model, policy = load_prism_model(stormpy.examples.files.prism_mdp_coin_2_2,
#                                   "Pmin=? [F \"finished\" & \"all_coins_equal_1\"]")
# TERM_LABELS = ('finished')

# model, policy = load_prism_model(stormpy.examples.files.prism_dtmc_die)
# TERM_LABELS = ('done')

# path = '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/mdp/slipgrid.nm'
# formula = "Rmin=? [F \"goal\"]"
# TERM_LABELS = ('goal')

# path = '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/dtmc/brp-16-2.pm'
# formula = None
# TERM_LABELS = ('target')

# Load PRISM model with STORM
# model, policy = load_prism_model(path, formula)

path = '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/dtmc/dummy.nm'
TERM_LABELS = ('done')
model, policy = load_prism_model(path)

uncertainty_model = L1_polytope
L1_size     = 0.000001

# Parse model and policy
M = parse_storm(model, policy, uncertainty_model, L1_size, TERM_LABELS)

# Hyperparameters for optimization problem
M.DISCOUNT          = 0.9
M.ALPHA_PENALTY     = 0
M.DELTA     	    = 1e-6

# %%

PI = parse_policy(M, policy)
PI = generate_random_policy(M)

# Verify loaded model by solving the optimization program
CVX = verify_cvx(M, PI, verbose=True)
CVX.solve(solver=SOLVER, store_initial = True)

print('\nValue of the measure in initial state:', CVX.prob.value)

# Check if complementary slackness is satisfied
CVX.check_complementary_slackness()

# %%    

SVT = gradients_cvx(M, PI, CVX.x, CVX.alpha, CVX.beta, CVX.cns)

for THETA_SA,THETA in M.parameters.items():

    tocDiff(False)
    Vx = SVT.solve(M, PI, THETA, 'SCS')
    print('\nParameter {} solved in {} seconds'.format(THETA_SA, tocDiff(False)))
    print('Gradient in initial state:', Vx[list(M.sI.keys())].value)
    
    grad_analytical   = Vx.value[0:len(M.states)]
    
    cum_diff = CVX.delta_solve(THETA, M.DELTA, grad_analytical, SOLVER, verbose = False)
    
    print('- Cum.diff. between analytical and numerical is {:.3f}'.format(cum_diff))