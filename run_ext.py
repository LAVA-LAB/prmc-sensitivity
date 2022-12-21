# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:05:56 2022

@author: Thom Badings
"""

# %run "~/documents/sensitivity-prmdps/prmdp-sensitivity-git/run_ext.py"

import numpy as np

import stormpy
import stormpy.examples
import stormpy.examples.files

from core.cvx import verify_cvx

def load_prism_model(path, formula):
    
    program = stormpy.parse_prism_program(path)
    formulas = stormpy.parse_properties(formula, program)
    
    model = stormpy.build_model(program)

    if model.is_nondeterministic_model:
        
        result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)
        policy = result.scheduler

    else:
        
        policy = None

    return model, policy

model, policy = load_prism_model(stormpy.examples.files.prism_mdp_coin_2_2,
                                 "Pmin=? [F \"finished\" & \"all_coins_equal_1\"]")


# %%

from models.parse_model import parse_storm
from models.uncertainty_models import L0_polytope, L1_polytope

uncertainty_model = L1_polytope
L1_size     = 0.0001

M = parse_storm(model, uncertainty_model, L1_size)

THETA_SA = (0,0)
THETA = M.parameters[THETA_SA]
SOLVER = 'GUROBI'
DELTA = 1e-4

cvx_prob, cns, x, alpha, beta = verify_cvx(M, policy, verbose=True, solver=SOLVER)
print('\nOptimal value:', cvx_prob.value)

# Check if assumption is satisfied
# for s,state in M.graph.items():
#     for a,action in state.items():
#         assert np.all(np.sum([
#                     np.abs(cns[('ineq',s,a)].dual_value) < 1e-9,
#                     np.abs(alpha[(s,a)].value) < 1e-9
#                 ], axis=0) == 1)

# %%

import pandas as pd

from core.sensitivity import sensitivity_jacobians_full, sensitivity_cvx_sparse
gradients, Dgx, Dgv = sensitivity_jacobians_full(M, policy, x, alpha, beta, cns, THETA)

sensitivity_cvx_sparse(M, policy, x, alpha, beta, cns, THETA)

THETA.value += DELTA

_, _, x_delta, _, _ = verify_cvx(M, policy, verbose=False)

analytical   = gradients[0:len(M.states)]
experimental = (x_delta.value-x.value)/DELTA

dct = {'analytical': analytical,
       'experimental': experimental,
       'abs.diff.': np.round(analytical - experimental, 4)}

results = pd.DataFrame(dct)

print('\nPartial derivatives for parameter {}:\n'.format(THETA_SA))
print(results)

print('\nGradients from CVXPY: ', THETA.gradient)
print('\nJacobian Dgx condition number: ', np.linalg.cond(Dgx))
