# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:22:11 2022

@author: Thom Badings
"""

import numpy as np
import scipy.linalg as linalg
from scipy.sparse import coo_matrix, bmat, diags, identity, block_diag, csc_matrix, csr_matrix
import cvxpy as cp

from core.commons import tocDiff
from core.commons import unit_vector, valuate, deriv_valuate

def gradients_spsolve(M, CVX, verbose = False):
    
    tocDiff(False)
    
    x = CVX.x
    alpha = CVX.alpha
    beta = CVX.beta
    cns = CVX.cns
    
    Da_f = -identity(M.robust_constraints)
    LaDa_f = diags(np.concatenate([cns[('ineq',s,a)].dual_value for (s,a) in M.robust_pairs_suc.keys()]))
    diag_f = diags(np.concatenate([alpha[(s,a)].value for (s,a) in M.robust_pairs_suc.keys()]))

    A11 = np.eye(len(M.states))
    for s in M.states:
      if not s.terminal:
        for a in s.actions:
            if not a.robust:                
                for ss, p_hat in zip(a.successors, a.model.probabilities):
                    A11[s.id, ss] = -M.DISCOUNT * s.policy[a.id] * p_hat
                
    A21 = np.zeros(( M.robust_successors, len(M.states) ))
    i = 0
    for (s_id,a_id) in M.robust_pairs_suc.keys():
        succ = M.states_dict[s_id].actions_dict[a_id].successors
        for ss_id in succ:
            A21[i, ss_id] = 1
            i += 1

    A12 = np.zeros(( len(M.states), M.robust_constraints ))
    A13 = np.zeros(( len(M.states), len(M.robust_pairs_suc) ))
    for i, (s_id,a_id) in enumerate(M.robust_pairs_suc.keys()):
        s = M.states_dict[s_id]
        
        j = s.actions_dict[a_id].alpha_start_idx
        b = s.actions_dict[a_id].model.b

        A12[s_id, j:j+len(b)] = M.DISCOUNT * s.policy[a_id] * valuate(b)
            
        A13[s_id, i] = M.DISCOUNT * s.policy[a_id]

    A22 = block_diag([M.states_dict[s_id].actions_dict[a_id].model.A.T for (s_id,a_id) in M.robust_pairs_suc.keys() ])

    A23 = block_diag([np.ones(( n, 1 )) for n in M.robust_pairs_suc.values() ])

    J = bmat([[ None, None,   None, None,   A11.T, A21.T ],
              [ None, None,   None, Da_f,   A12.T, A22.T ],
              [ None, None,   None, None,   A13.T, A23.T ],
              [ None, LaDa_f, None, diag_f, None,  None  ],
              [ A11,  A12,    A13,  None,   None,  None  ],
              [ A21,  A22,    A23,  None,   None,  None  ]], format='csr')

    #####

    # Sparse matrix creation
    tocDiff(False)

    iters = 2*sum([len(M.param2stateAction[th]) for th in M.parameters.values()])
    row = [[] for i in range(iters)]
    col = [[] for i in range(iters)]
    data = [[] for i in range(iters)]
    i = 0

    offset_A = len(M.states)
    offset_B = len(M.states) + len(M.robust_pairs_suc) + 2*M.robust_constraints
    full_rows = 2*len(M.states) + len(M.robust_pairs_suc) + 2*M.robust_constraints + M.robust_successors

    for v, (THETA_SA,THETA) in enumerate(M.parameters.items()):
        for (s_id,a_id) in M.param2stateAction[THETA]:
            
            j = M.states_dict[s_id].actions_dict[a_id].alpha_start_idx
            b = M.states_dict[s_id].actions_dict[a_id].model.b
            s = M.states_dict[s_id]
            
            row[i]  = np.arange(j, j+len(b)) + offset_A
            col[i]  = np.ones(len(b)) * v
            data[i] = M.DISCOUNT * s.policy[a_id] * deriv_valuate(b, THETA) * cns[s_id].dual_value

            row[i+1]  = np.array([s_id + offset_B])
            col[i+1]  = np.array([v])
            data[i+1] = np.array([M.DISCOUNT * s.policy[a_id] * deriv_valuate(b, THETA) @ alpha[(s_id, a_id)].value])
            
            i += 2

    Ju_COO = csc_matrix((np.concatenate(data), (np.concatenate(row), np.concatenate(col))), shape=(full_rows, len(M.parameters)))

    print('Sparse matrix creation took: {}'.format(tocDiff(False)))

    # # Nonsparse matrix creation
    # tocDiff(False)

    # DuVaL = np.zeros(( M.robust_constraints, len(M.parameters) ))
    # DuH = np.zeros(( len(M.states), len(M.parameters) ))

    # for v, (THETA_SA,THETA) in enumerate(M.parameters.items()):
    #     for (s_id,a_id) in M.param2stateAction[THETA]:
            
    #         j = M.states_dict[s_id].actions_dict[a_id].alpha_start_idx
    #         b = M.states_dict[s_id].actions_dict[a_id].model.b
    #         s = M.states_dict[s_id]
            
    #         DuVaL[j:j+len(b), v] = M.DISCOUNT * s.policy[a_id] * deriv_valuate(b, THETA) * CVX.cns[s_id].dual_value
                
    #         DuH[s_id, v] += M.DISCOUNT * s.policy[a_id] * deriv_valuate(b, THETA) @ CVX.alpha[(s_id, a_id)].value
          
    # Ju = np.vstack(( np.zeros((len(M.states), len(M.parameters))),
    #                   -DuVaL,
    #                   np.zeros((len(M.robust_pairs_suc), len(M.parameters))),
    #                   np.zeros((M.robust_constraints, len(M.parameters))),
    #                   -DuH,
    #                   np.zeros(( M.robust_successors, len(M.parameters)))
    #                   ))  
          
    # print('Non-sparse matrix creation took: {}'.format(tocDiff(False)))  
       
    from scipy.sparse.linalg import spsolve

    print('Solve linear equation system...')

    tocDiff(False)
    GRAD = spsolve(J, Ju_COO)
    #print('\nLinear equation system for {} parameters solved in {} seconds'.format(len(M.parameters), tocDiff(False)))

    # tocDiff(False)
    # GRAD = spsolve(J, Ju)[0,:]
    # print('\nLinear equation system for {} parameters solved in {} seconds'.format(len(M.parameters), tocDiff(False)))

    # tocDiff(False)
    # GRAD = np.linalg.solve(J.toarray(), Ju_COO.toarray())
    # print('\nLinear equation system for {} parameters solved in {} seconds'.format(len(M.parameters), tocDiff(False)))

    return GRAD

class gradients_cvx:
    
    def __init__(self, M, x, alpha, beta, cns, verbose = False):
        
        print('\nDefine optimization problem to compute gradients...')
        
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
                for a in s.actions:
                    if a.robust:
                        self.Valpha[(s.id, a.id)] = cp.Variable(len(a.model.b))
                        self.Vbeta[(s.id, a.id)]  = cp.Variable()
                        
                        self.Vlambda[(s.id, a.id)] = cp.Variable(len(a.model.b))
                        self.Vnu_dual[(s.id, a.id)] = cp.Variable(len(a.model.A.T))
                        
                        self.Dtheta_Nabla_L[(s.id, a.id)] = cp.Parameter(len(a.model.b))
        
        for s in M.states:
            if verbose:
                print('\nAdd for state {}'.format(s.id))
            
            # 1) Dx h(y,theta).T @ Nabla V^nu = 0
            # For each state: Vnu_primal of that state, plus the sum of all Vnu_dual
            # related to that state, equals zero.
            
            SUM = 0
            for (s_pre, a_id, c) in M.poly_pre_state[s.id]:
                a = M.states_dict[s_pre].actions_dict[a_id]
                    
                SUM += self.Vnu_dual[(s_pre, a_id)][c]
                    
            for (s_pre, a_id, p) in M.distr_pre_state[s.id]: 
                   
                SUM -= M.DISCOUNT * M.states_dict[s_pre].policy[a_id] * p * self.Vnu_primal[s_pre]
                    
            self.Vcns[('g1', s.id)] = self.Vnu_primal[s.id] + SUM == 0
            
            if not s.terminal:
              for a in s.actions:
                if a.robust:
                    # 2)
                    # Add a vector constraint for each state-action pair
                    
                    # For each robust state-action pair: minus Vlambda, plus the
                    # probability of choosing action a times b vector times Vnu_primal,
                    # plus the valae of the A matrix times Vnu_dual, equals the
                    # derivative of that whole thing wrt the parameter
                    self.Vcns[('g2', s.id, a.id)] = \
                        - self.Vlambda[(s.id, a.id)] \
                        + M.DISCOUNT * s.policy[a.id] * valuate(a.model.b) * self.Vnu_primal[s.id] \
                        + valuate(a.model.A) @ self.Vnu_dual[(s.id, a.id)] == - self.Dtheta_Nabla_L[(s.id, a.id)]
                        
                    # 3)
                    # Add a scalar constraint for each robust state-action pair
                    
                    # For each robust state-action pair: the probability of choosing
                    # action a times the Vnu_primal variable, plus the sum of Vnu_dual
                    # over all places where it occurs, is zero.
                    self.Vcns[('g3', s.id, a.id)] = \
                        M.DISCOUNT * s.policy[a.id] * self.Vnu_primal[s.id] + cp.sum(self.Vnu_dual[(s.id, a.id)]) == 0
                
                    # 4) For each lambda / alpha vector
                    
                    # Lambda-tilde times Valpha == alpha* times Vlambda
                    lambda_alpha = cp.multiply(self.cns[('ineq', s.id, a.id)].dual_value, self.Valpha[(s.id, a.id)])
                    alpha_lambda = cp.multiply(self.alpha[(s.id, a.id)].value, self.Vlambda[(s.id, a.id)])
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
                if verbose:
                    print('-- State {} is terminal'.format(s.id))
                
                # If state is terminal, sensitivity is zero
                self.Vcns[('g5a', s.id)] = \
                    self.Vx[s.id] == 0
            
            else:
                SUM = 0
                
                # For each action in the policy at this state
                for a in s.actions:
                    
                    if verbose:
                        print('-- Add action {} with probability {:.3f}'.format(a.id, s.policy[a.id]))
                    
                    if a.robust:
                        
                        if verbose:
                            print('--- Robust action')
                        SUM += s.policy[a.id] * (valuate(a.model.b) @ self.Valpha[(s.id, a.id)] + self.Vbeta[(s.id, a.id)])
                
                    else:
                        
                        if verbose:
                            print('--- Nonrobust action')
                        SUM -= s.policy[a.id] * (a.model.probabilities @ self.Vx[a.model.states])
                        
                self.Vcns[('g5a', s.id)] = self.Vx[s.id] + M.DISCOUNT*SUM == -self.Dtheta_h[s.id]
                    
        print('Build optimization problem...')
        self.sens_prob = cp.Problem(objective = cp.Minimize(0), constraints = self.Vcns.values())
              
        if self.sens_prob.is_dcp(dpp=True):
            print('Program satisfies DCP rule set')
        else:
            print('Program does not satisfy DCP rule set')
    
    def solve(self, M, theta, solver = 'SCS'):
        
        print('\nSet parameter values...')
        
        # Set entries depending on the parameter theta
        Dtheta_h = np.zeros(len(M.states))
        
        for s in M.states:
            if not s.terminal:
                
                for a in s.actions:                    
                    if a.robust:
                        Dtheta_h[s.id] += M.DISCOUNT * s.policy[a.id] * deriv_valuate(a.model.b, theta) @ self.alpha[(s.id, a.id)].value
                    
                        self.Dtheta_Nabla_L[(s.id, a.id)].value = \
                            M.DISCOUNT * s.policy[a.id] * deriv_valuate(a.model.b, theta) * self.cns[s.id].dual_value
              
        self.Dtheta_h.value = Dtheta_h          
             
        print('\nSolve sensitivity program...')
        
        # Solve optimization problem
        if solver == 'GUROBI':
            self.sens_prob.solve(solver='GUROBI')
        else:
            self.sens_prob.solve(solver='SCS')
            
        print('Status of computing gradients:', self.sens_prob.status)
        
        return self.Vx