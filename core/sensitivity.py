# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:22:11 2022

@author: Thom Badings
"""

import numpy as np
import scipy.linalg as linalg
import cvxpy as cp

from core.commons import unit_vector, valuate, deriv_valuate

def sensitivity_jacobians_full(M, policy, x, alpha, beta, cns, THETA):

    A_elems = []
    dA_elems = []
    C_rows = []
    D_elems = []
    E_cols = []
    
    for s in M.states:
        # If state s is not terminal
        if s not in M.states_term:
            A_elems += [list(policy[s][a] * valuate(action['b'])) for a,action in M.graph[s].items() ]
            dA_elems += [list(policy[s][a] * deriv_valuate(action['b'], THETA)) for a,action in M.graph[s].items() ]
            
            for action in M.graph[s].values():
                for succ in action['succ']:
                    C_rows += [unit_vector(len(M.states), succ)]
                D_elems += [valuate(action['A']).T]
                E_cols += [np.ones((len(action['succ']), 1))]
        else:
            A_elems += [[]]
            dA_elems += [[]]
    
    B = np.zeros((len(M.states), len(beta)))
    cl = 0
    for s,state in M.graph.items():    
        if s not in M.states_term:
            B[s, cl] = np.sum([policy[s][a] for a,action in state.items()])
            cl += 1
    
    # B = np.array([[np.sum([policy[s][a]*beta[(s,a)].value for a,action in state.items()]) for s,state in M.graph.items()]]).T
    
    A = np.block([
            [np.eye(len(M.states)),  linalg.block_diag(*A_elems), B],
            [np.array(C_rows),  linalg.block_diag(*D_elems), linalg.block_diag(*E_cols)]
        ])
    
    alphaSum = sum(a.size for a in alpha.values())
    G = np.block([
            [np.zeros((alphaSum, len(M.states))), -np.eye(alphaSum), np.zeros((alphaSum, len(beta)))]
        ])
    
    lambda_flat = np.array([cns[('ineq',s,a)].dual_value for s,state in M.graph.items() for a,action in state.items() if ('ineq',s,a) in cns]).flatten()
    nu_flat     = np.concatenate([
            np.array([cns[(s)].dual_value for s in M.states]).flatten(),
            np.array([cns[(s,a)].dual_value  for s,state in M.graph.items() 
                      for a,action in state.items() if (s,a) in cns]).flatten()
        ])
    
    decvar_flat = np.concatenate([
                    x.value,
                    np.concatenate([alpha[(s,a)].value for s,state in M.graph.items() for a,action in state.items()]),
                    np.array([beta[(s,a)].value for s,state in M.graph.items() for a,action in state.items()]),
                    ])
    
    Dgx = np.block([
            [np.zeros((G.T.shape[0], G.T.shape[0])),    G.T,                                    A.T],
            [np.diag(lambda_flat) @ G,                  np.diag(G @ decvar_flat),               np.zeros((len(lambda_flat), A.T.shape[1]))],
            [A,                                         np.zeros((A.shape[0], G.T.shape[1])),   np.zeros((A.shape[0], A.T.shape[1]))]
        ])
    
    dA = np.block([
            [np.zeros((len(M.states),len(M.states))),     linalg.block_diag(*dA_elems),                   np.zeros(B.shape)],
            [np.zeros(np.array(C_rows).shape),  np.zeros(linalg.block_diag(*D_elems).shape),    np.zeros(linalg.block_diag(*E_cols).shape)]
        ])
    
    Dgv = np.concatenate([
            dA.T @ nu_flat,
            np.zeros(len(lambda_flat)),
            dA @ decvar_flat
        ])
    
    gradients = np.linalg.solve(Dgx, -Dgv)
    
    return gradients, Dgx, Dgv

def sensitivity_cvx_sparse(M, policy, x, alpha, beta, cns, THETA):
    
    