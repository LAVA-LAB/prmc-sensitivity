# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:54:32 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np
import numpy.polynomial.polynomial as poly
import copy

def solve_LP(states, edges, states_post, states_pre, goal, verbose=False):
    # Sparse implementation of reachability computation via LP
    
    if verbose:
        print('\nSolve LP for expected reward...')
    
    sp_x = cp.Variable(len(states), nonneg=True)
    sp_aLow = {(s,ss): cp.Variable() for (s,ss) in edges.keys()}
    sp_aUpp = {(s,ss): cp.Variable() for (s,ss) in edges.keys()}
    sp_beta = {s: cp.Variable() for s in states_nonterm}
    
    constraints = {}
    
    for s in states:
        if verbose:
            print(' - Add constraints for state', s)
    
        constraints[('nu',s)] = sp_x[s] == (s in goal) \
                + cp.sum([sp_aLow[(s,ss)]*edges[(s,ss)][0].val() for ss in states_post[s]]) \
                - cp.sum([sp_aUpp[(s,ss)]*edges[(s,ss)][1].val() for ss in states_post[s]]) \
                - (sp_beta[s] if s in states_nonterm else 0)
        
    for (s,ss),e in edges.items():
        if verbose:
            print(' - Add edge from',s,'to',ss)
    
        constraints[('la_low',s,ss)] = sp_aLow[(s,ss)] >= 0 
        constraints[('la_upp',s,ss)] = sp_aUpp[(s,ss)] >= 0
            
        constraints[('mu',s,ss)] = sp_aUpp[(s,ss)] - sp_aLow[(s,ss)] + sp_x[ss] + sp_beta[s] == 0
        
    # Concatenate all constraints into a list
    sp_prob = cp.Problem(objective = cp.Maximize(sp_x[sI]), constraints = constraints.values())
    
    if verbose:
        print('Is problem DCP?', sp_prob.is_dcp(dpp=True))
    
    sp_prob.solve()
    
    if verbose:
        print('Status:', sp_prob.status)
    
    return constraints, sp_x, sp_aLow, sp_aUpp

def sensitivity_LP(states, edges, states_post, states_pre, constraints, aLow, aUpp, verbose=False):
    # Perform sensitivity analysis
    
    if verbose:
        print('\nDefine system of equations for sensitivity analysis...')
    
    Dth_x = cp.Variable(len(states))
    Dth_aLow = {(s,ss): cp.Variable() for (s,ss) in edges.keys()}
    Dth_aUpp = {(s,ss): cp.Variable() for (s,ss) in edges.keys()}
    Dth_beta = {s: cp.Variable() for s in states_nonterm}
    
    Dth_nu = cp.Variable(len(states))
    Dth_mu = {(s,ss): cp.Variable() for (s,ss) in edges.keys()}
    
    GC = []
    
    # Differentiate lower bound from 0 to 1
    X = {}
    Y = {}
    Z = {}
    
    # Enumerate over states
    for s in states:
        if verbose:
            print(' - Add constraints for state', s)
        
        if s in states_nonterm:
            beta_sub = Dth_beta[s]
            
            # 3
            GC += [Dth_nu[s] + cp.sum([Dth_mu[(s,ss)] for ss in states_post[s]]) == 0]
        else:
            beta_sub = 0
            
        # 1
        GC += [Dth_nu[s] + cp.sum([Dth_mu[(ss,s)] for ss in states_pre[s]]) == 0]
        
        # 5
        X[(s)] = cp.Parameter()
        GC += [Dth_x[s] + cp.sum([- edges[(s,ss)][0].val()*Dth_aLow[(s,ss)]
                                  + edges[(s,ss)][1].val()*Dth_aUpp[(s,ss)] 
                                  for ss in states_post[s]])
                        + beta_sub == -X[(s)]]
        
    # Enumerate over edges
    for (s,ss),e in edges.items():
        if verbose:
            print(' - Add edge from',s,'to',ss)
        
        Y[(s,ss)] = cp.Parameter()
        Z[(s,ss)] = cp.Parameter()
        
        GC += [ 
            # 2 + 4
            constraints[('la_low',s,ss)].dual_value / aLow[(s,ss)].value * Dth_aLow[(s,ss)] == 
                ( e[0].val()*Dth_nu[s] + Dth_mu[(s,ss)] - Y[(s,ss)] ),
            constraints[('la_upp',s,ss)].dual_value / aUpp[(s,ss)].value * Dth_aUpp[(s,ss)] == 
                ( - e[1].val()*Dth_nu[s] - Dth_mu[(s,ss)] + Z[(s,ss)] ),
            # 6
            Dth_x[ss] - Dth_aLow[(s,ss)] + Dth_aUpp[(s,ss)] + Dth_beta[s] == 0
            ]
        
    Dth_prob = cp.Problem(cp.Maximize(0), constraints = GC)
        
    return Dth_prob, Dth_x, X, Y, Z

# %%

class pol(object):
    
    def __init__(self, param, coeff):
        
        self.coeff = coeff
        self.par  = param
    
    def deriv_eval(self, param):
        # Differentiate polynomial and return the value evaluated at the 
        # current parameter value
        
        # Check if ID of the provided parameter equals that of this polynomial
        if param.id == self.par.id:
        
            coeff_der = poly.polyder(self.coeff)
    
            return cp.sum([c * self.par ** i for i,c in enumerate(coeff_der)]).value
    
        # If the IDs don't match, then the derivative is zero by default
        else:
            return 0
    
    def expr(self):
        # Evaluate the polynomial to get the CVXPY expression
        
        return cp.sum([c * self.par ** i for i,c in enumerate(self.coeff)])
    
    def val(self):
        # Evaluate the polynomial and return the value evaluated at the 
        # current parameter value
        
        return cp.sum([c * self.par.value ** i for i,c in enumerate(self.coeff)])
        

params = {
    'par1': cp.Parameter(value = 0.31),
    'par2': cp.Parameter(value = 0.9),
    'par3': cp.Parameter(value = 0.2),
    'par4': cp.Parameter(value = 0.8),
    'par5': cp.Parameter(value = 0.2),
    'par6': cp.Parameter(value = 0.3),
    #
    'par7': cp.Parameter(value = 0.3),
    'par8': cp.Parameter(value = 0.6),
    'par9': cp.Parameter(value = 0.35),
    'par10': cp.Parameter(value = 0.6),
    }

states = set({0,1,2,3})

edges = {
    (0,1): [pol(params['par1'], [0, 0, 1]), pol(params['par2'], [0, 1])],
    (0,2): [pol(params['par3'], [0, 1]), pol(params['par4'], [0, 1])],
    (0,3): [pol(params['par5'], [0, 1]), pol(params['par6'], [0, 1])],
    (2,0): [pol(params['par7'], [0, 1]), pol(params['par8'], [0, 1])],
    (2,3): [pol(params['par9'], [0, 1]), pol(params['par10'], [0, 1])],
    }

edges_fix = copy.deepcopy(edges)

states_post = {s: [ss for ss in states if (s,ss) in edges] for s in states}
states_pre  = {s: [ss for ss in states if (ss,s) in edges] for s in states}

# Initial state index
goal = set({1})
absorbing = set({3})
sI = 0

states_nonterm = states - goal - absorbing

# %%

delta = 1e-6

# Solve initial LP
constraints, x, aLow, aUpp = solve_LP(states, edges, states_post, states_pre, goal)

x_orig = copy.deepcopy(x)

print('Reward in sI:', np.round(x_orig[sI].value, 8))

# Setup sensitivity LP
Dth_prob, Dth_x, X, Y, Z = sensitivity_LP(states, edges_fix, states_post, states_pre, constraints, aLow, aUpp)

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
    
    _, x_delta, _, _ = solve_LP(states, edges, states_post, states_pre, goal)
    
    numerical = np.round((x_delta[sI].value - x_orig[sI].value) / delta, 6)
    
    param.value -= delta
    
    diff = np.round(analytical - numerical, 4)
    results.loc[key] = [analytical, numerical, diff]
    
print(results)