# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:54:32 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np
import copy

from cvx import solve_LP, sensitivity_LP

# %%
    
params = {
    'par1': cp.Parameter(value = 0.3),
    'par2': cp.Parameter(value = 0.9),
    'par3': cp.Parameter(value = 0.2),
    'par4': cp.Parameter(value = 0.8),
    #
    'par5': cp.Parameter(value = 0.2),
    'par6': cp.Parameter(value = 0.5),
    'par7': cp.Parameter(value = 0.3),
    'par8': cp.Parameter(value = 0.6),
    #
    'par9': cp.Parameter(value = 0.35),
    'par10': cp.Parameter(value = 1.01),
    #
    'par11': cp.Parameter(value = 0.35),
    'par12': cp.Parameter(value = 1.01),
    }

states = set({0,1,2,3,4})

edges = {
    (0,1): [pol(params['par1'], [0, 1]), pol(params['par2'], [0, 1])],
    (0,2): [pol(params['par3'], [0, 1]), pol(params['par4'], [0, 1])],
    (1,2): [pol(params['par5'], [0, 1]), pol(params['par6'], [0, 1])],
    (1,3): [pol(params['par7'], [0, 1]), pol(params['par8'], [0, 1])],
    (2,4): [pol(params['par9'], [0, 1]), pol(params['par10'], [0, 1])],
    (3,4): [pol(params['par11'], [0, 1]), pol(params['par12'], [0, 1])],
    }

edges_fix = copy.deepcopy(edges)

states_post = {s: [ss for ss in states if (s,ss) in edges] for s in states}
states_pre  = {s: [ss for ss in states if (ss,s) in edges] for s in states}

# Initial state index
reward = [0,1,2,3,0]

terminal = set({4})
sI = 0

states_nonterm = states - terminal

# %%

delta = 1e-6

# Solve initial LP
constraints, x, aLow, aUpp = solve_LP(states, sI, edges, states_post, states_pre, states_nonterm, reward)

x_orig = copy.deepcopy(x)

print('Reward in sI:', np.round(x_orig[sI].value, 8))

# Setup sensitivity LP
Dth_prob, Dth_x, X, Y, Z = sensitivity_LP(states, edges_fix, states_post, states_pre, states_nonterm,
                                          constraints, aLow, aUpp)

# %%

# Define for which parameter to differentiate        

import pandas as pd

results = pd.DataFrame(columns = ['analytical', 'numerical', 'abs.diff.'])

for key, param in params.items():

    for s in states:
        
        X[(s)].value = cp.sum([
                        - aLow[(s,ss)].value * edges[(s,ss)][0].deriv_eval(param)
                        + aUpp[(s,ss)].value * edges[(s,ss)][1].deriv_eval(param)        
                        for ss in states_post[s]])
        
        # if s != s_from:
        #     X[(s)].value = 0
        # else:
        #     if bound == 'low':
        #         X[(s)].value = -aLow[(s,s_to)].value * edges[(s,s_to)][0].deriv_eval()
        #     else:
        #         X[(s)].value = aUpp[(s,s_to)].value * edges[(s,s_to)][1].deriv_eval()
    
    for (s,ss),e in edges.items():
        
        Y[(s,ss)].value = -constraints[('nu',s)].dual_value * edges[(s,ss)][0].deriv_eval(param)
        Z[(s,ss)].value =  constraints[('nu',s)].dual_value * edges[(s,ss)][1].deriv_eval(param)
        
        # if s != s_from or ss != s_to:
        #     Y[(s,ss)].value = 0
        #     Z[(s,ss)].value = 0
        # else:
        #     if bound == 'low':
        #         Y[(s,ss)].value = -constraints[('nu',s)].dual_value * edges[(s,s_to)][0].deriv_eval()
        #         Z[(s,ss)].value = 0
        #     else:
        #         Y[(s,ss)].value = 0
        #         Z[(s,ss)].value = constraints[('nu',s)].dual_value * edges[(s,s_to)][1].deriv_eval()
                    
    Dth_prob.solve()
    
    analytical = np.round(Dth_x[sI].value, 6)
    
    param.value += delta
    
    _, x_delta, _, _ = solve_LP(states, sI, edges, states_post, states_pre, states_nonterm, reward)
    
    numerical = np.round((x_delta[sI].value - x_orig[sI].value) / delta, 6)
    
    param.value -= delta
    
    diff = np.round(analytical - numerical, 4)
    results.loc[key] = [analytical, numerical, diff]
    
print(results)