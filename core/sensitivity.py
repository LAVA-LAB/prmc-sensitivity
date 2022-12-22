# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:22:11 2022

@author: Thom Badings
"""

import numpy as np
import scipy.linalg as linalg
import cvxpy as cp

from core.commons import tocDiff
from core.commons import unit_vector, valuate, deriv_valuate

class gradients_cvx:
    
    def __init__(self, M, PI, x, alpha, beta, cns):
        
        # Store variables in object
        self.x          = x
        self.alpha      = alpha
        self.beta       = beta
        self.cns        = cns
        
        # Define primal decision variables
        self.Vx          = cp.Variable(len(M.states))
        self.Valpha      = {}
        self.Vbeta       = {}
        
        # Define dual decision variables
        self.Vlambda     = {}
        self.Vnu_primal  = cp.Variable(len(M.states))
        self.Vnu_dual    = {}
        
        # Define parameters for the RHS of the system of equations        
        self.Dtheta_Nabla_L = {}
        self.Dtheta_h       = cp.Parameter(len(M.states))
        
        self.Vcns = {}
        
        for s in M.states: 
            if not s.terminal:
                for a,prob in PI[s.id].items():
                    if a.robust:
                        self.Valpha[(s.id, a.id)] = cp.Variable(len(a.model.b))
                        self.Vbeta[(s.id, a.id)]  = cp.Variable()
                        
                        self.Vlambda[(s.id, a.id)] = cp.Variable(len(a.model.b))
                        self.Vnu_dual[(s.id, a.id)] = cp.Variable(len(a.model.A.T))
                        
                        self.Dtheta_Nabla_L[(s.id, a.id)] = cp.Parameter(len(a.model.b))
        
        for s in M.states:        
            print('\nAdd for state {}'.format(s.id))
            
            # 1) Dx h(y,theta).T @ Nabla V^nu = 0
            # For each state: Vnu_primal of that state, plus the sum of all Vnu_dual
            # related to that state, equals zero.
            
            SUM = 0
            for (s_pre, a_id, c) in M.poly_pre_state[s.id]:
                a = M.states_dict[s_pre].actions_dict[a_id]
                if a in PI[s_pre]:
                    
                    SUM += self.Vnu_dual[(s_pre, a_id)][c]
                    
            for (s_pre, a_id, p) in M.distr_pre_state[s.id]:
                if M.states_dict[s_pre].actions_dict[a_id] in PI[s_pre]:
                    
                    a = M.states_dict[s_pre].actions_dict[a_id]
                    SUM -= PI[s_pre][a] * p * self.Vnu_primal[s_pre]
                    
            self.Vcns[('g1', s.id)] = self.Vnu_primal[s.id] + SUM == 0
            
            if not s.terminal:
              for a,prob in PI[s.id].items():            
                if a.robust:
                    # 2)
                    # Add a vector constraint for each state-action pair
                    
                    # For each robust state-action pair: minus Vlambda, plus the
                    # probability of choosing action a times b vector times Vnu_primal,
                    # plus the valae of the A matrix times Vnu_dual, equals the
                    # derivative of that whole thing wrt the parameter
                    self.Vcns[('g2', s.id, a.id)] = \
                        - self.Vlambda[(s.id, a.id)] \
                        + prob * valuate(a.model.b) * self.Vnu_primal[s.id] \
                        + valuate(a.model.A) @ self.Vnu_dual[(s.id, a.id)] == - self.Dtheta_Nabla_L[(s.id, a.id)]
                        
                    # 3)
                    # Add a scalar constraint for each robust state-action pair
                    
                    # For each robust state-action pair: the probability of choosing
                    # action a times the Vnu_primal variable, plus the sum of Vnu_dual
                    # over all places where it occurs, is zero.
                    self.Vcns[('g3', s.id, a.id)] = \
                        prob * self.Vnu_primal[s.id] + cp.sum(self.Vnu_dual[(s.id, a.id)]) == 0
                
                    # 4) For each lambda / alpha vector
                    
                    # Lambda-tilde times Valpha == alpha* times Vlambda
                    lambda_alpha = cp.multiply(cns[('ineq', s.id, a.id)].dual_value, self.Valpha[(s.id, a.id)])
                    alpha_lambda = cp.multiply(alpha[(s.id, a.id)].value, self.Vlambda[(s.id, a.id)])
                    self.Vcns[('g4', s.id, a.id)] = (lambda_alpha == alpha_lambda)
                        
                    del lambda_alpha
                    del alpha_lambda
                    
                    # 5b) For each dual equality constraint, replace original decision
                    # variables with the ones of the sensitivity problem
                    self.Vcns[('g5b', s.id, a.id)] = \
                        a.model.A.T @ self.Valpha[(s.id, a.id)] + self.Vx[a.successors] + self.Vbeta[(s.id, a.id)] == 0
                        
            # 5a) For each reward equality constraint, replace original decision
            # variables with the ones of the sensitivity problem
            if s.terminal:
                print('-- State {} is terminal'.format(s.id))
                
                # If state is terminal, sensitivity is zero
                self.Vcns[('g5a', s.id)] = \
                    self.Vx[s.id] == 0
            
            else:
                SUM = 0
                
                # For each action in the policy at this state
                for a, prob in PI[s.id].items():
                    
                    print('-- Add action {} with probability {:.3f}'.format(a.id, prob))
                    
                    if a.robust:
                        
                        print('--- Robust action')
                        SUM += prob * (valuate(a.model.b) @ self.Valpha[(s.id, a.id)] + self.Vbeta[(s.id, a.id)])
                
                    else:
                        
                        print('--- Nonrobust action')
                        SUM -= prob * (a.model.probabilities @ self.Vx[a.model.states])
                        
                self.Vcns[('g5a', s.id)] = self.Vx[s.id] + SUM == -self.Dtheta_h[s.id]
                    
        self.sens_prob = cp.Problem(objective = cp.Minimize(0), constraints = self.Vcns.values())
              
        if self.sens_prob.is_dcp(dpp=True):
            print('Program satisfies DCP rule set')
        else:
            print('Program does not satisfy DCP rule set')
    
    def solve(self, M, PI, theta, solver = 'SCS'):
        
        # Set entries depending on the parameter theta
        Dtheta_h = np.zeros(len(M.states))
        
        for s in M.states:
            if not s.terminal:
                
                for a, prob in PI[s.id].items():
                    
                    if a.robust:
                        Dtheta_h[s.id] += prob * deriv_valuate(a.model.b, theta) @ self.alpha[(s.id, a.id)].value
                    
                        self.Dtheta_Nabla_L[(s.id, a.id)].value = \
                            prob * deriv_valuate(a.model.b, theta) * self.cns[s.id].dual_value
              
        self.Dtheta_h.value = Dtheta_h          
             
        tocDiff()
        
        # Solve optimization problem
        if solver == 'GUROBI':
            self.sens_prob.solve(solver='GUROBI')
        else:
            self.sens_prob.solve(solver='SCS')
            
        print('Status of computing gradients:', self.sens_prob.status)
        
        return self.Vx
    
    
'''    
def sensitivity_jacobians_full(M, policy, x, alpha, beta, cns, THETA):

    A_elems = []
    dA_elems = []
    C_rows = []
    D_elems = []
    E_cols = []
    
    for s in M.states:
        # If state s is not terminal
        if s not in M.states_term:
            
            A_elems += [a.model.b for a in s.actions ]
            dA_elems += [deriv_valuate(action['b'], THETA) for a,action in s.actions ]
            
            for a in s.actions:
                for succ in a.successors:
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
'''