# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:22:11 2022

@author: Thom Badings
"""

import numpy as np
from numpy.linalg import inv
import scipy.linalg as linalg
from scipy.sparse import coo_matrix, bmat, diags, identity, block_diag, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scikits.umfpack import spsolve as spsolve_umf
import cvxpy as cp

from core.commons import tocDiff
from core.commons import unit_vector, valuate, deriv_valuate
from core.commons import rrange

class gradient:

    def __init__(self, M, verbose = False):
        
        self.num_states = len(M.states)
        
        self.Da_f = -identity(M.robust_constraints, format='csc')
    
        self.A11 = np.eye(len(M.states))
        for s in M.states:
          if not s.terminal:
            for a in s.actions:
                if not a.robust:                
                    for ss, p_hat in zip(a.successors, a.model.probabilities):
                        self.A11[s.id, ss] = -M.discount * s.policy[a.id] * p_hat
            
        #TODO make this stuff more efficient:        
            
        self.A21 = np.zeros(( M.robust_successors, len(M.states) ))
        i = 0
        for (s_id,a_id) in M.robust_pairs_suc.keys():
            succ = M.states_dict[s_id].actions_dict[a_id].successors
            for ss_id in succ:
                self.A21[i, ss_id] = 1
                i += 1
    
        self.A12 = np.zeros(( len(M.states), M.robust_constraints ))
        self.A13 = np.zeros(( len(M.states), len(M.robust_pairs_suc) ))
        for i, (s_id,a_id) in enumerate(M.robust_pairs_suc.keys()):
            s = M.states_dict[s_id]
            
            j = s.actions_dict[a_id].alpha_start_idx
            b = s.actions_dict[a_id].model.b
    
            self.A12[s_id, j:j+len(b)] = M.discount * s.policy[a_id] * valuate(b)
                
            self.A13[s_id, i] = M.discount * s.policy[a_id]
    
        self.A22 = block_diag([M.states_dict[s_id].actions_dict[a_id].model.A.T for (s_id,a_id) in M.robust_pairs_suc.keys() ], format='csc')
        self.A23 = block_diag([np.ones(( n, 1 )) for n in M.robust_pairs_suc.values() ])
    
    def update(self, M, CVX, mode):
        
        # Completely remove dual variables lambda and nu
        if mode == 'remove_dual':
            
            self.J = bmat([[ self.A11,  self.A12[:, CVX.keepalpha],   self.A13 ],
                           [ self.A21,  self.A22[:, CVX.keepalpha],   self.A23 ]], format='csc')
        
        # Reduce by removing redundant alpha and lambda dual variables
        elif mode == 'reduce_dual':
            
            self.J = bmat([[ None,      None,                         None,     None,                           self.A11.T, self.A21.T ],
                           [ None,      None,                         None,     self.Da_f[:, CVX.keeplambda],   self.A12.T, self.A22.T ],
                           [ None,      None,                         None,     None,                           self.A13.T, self.A23.T ],
                           [ self.A11,  self.A12[:, CVX.keepalpha],   self.A13, None,                           None,       None  ],
                           [ self.A21,  self.A22[:, CVX.keepalpha],   self.A23, None,                           None,       None  ]], format='csc')
    
        # Solve naively for the full system of equations with all dual variables
        else:        
    
            LaDa_f = diags(np.concatenate([CVX.cns[('ineq',s,a)].dual_value for (s,a) in M.robust_pairs_suc.keys()]))
            diag_f = diags(np.concatenate([CVX.alpha[(s,a)].value for (s,a) in M.robust_pairs_suc.keys()]))        
    
            self.J = bmat([[ None,      None,      None,       None,      self.A11.T,   self.A21.T ],
                           [ None,      None,      None,       self.Da_f, self.A12.T,   self.A22.T ],
                           [ None,      None,      None,       None,      self.A13.T,   self.A23.T ],
                           [ None,      LaDa_f,    None,       diag_f,    None,         None  ],
                           [ self.A11,  self.A12,  self.A13,   None,      None,         None  ],
                           [ self.A21,  self.A22,  self.A23,   None,      None,         None  ]], format='csc')
        
        #####
    
        if mode == 'remove_dual':
            
            nr_rows = len(M.states) + M.robust_successors
            
            iters = sum([len(M.param2stateAction[th]) for th in M.parameters.values()])
            row = [[]]*iters
            col = [[]]*iters
            data = [[]]*iters
            i = 0
            
            for v, (THETA_SA,THETA) in enumerate(M.parameters.items()):
                for (s_id,a_id) in M.param2stateAction[THETA]:
                    
                    j = M.states_dict[s_id].actions_dict[a_id].alpha_start_idx
                    b = M.states_dict[s_id].actions_dict[a_id].model.b
                    s = M.states_dict[s_id]
                    
                    row[i]  = np.array([s_id])
                    col[i]  = np.array([v])
                    data[i] = np.array([M.discount * s.policy[a_id] * deriv_valuate(b, THETA) @ CVX.alpha[(s_id, a_id)].value])
                    
                    i += 1
            
        else:
            
            iters = 2*sum([len(M.param2stateAction[th]) for th in M.parameters.values()])
            row = [[]]*iters
            col = [[]]*iters
            data = [[]]*iters
            i = 0
            
            offset_A = len(M.states)
            
            if mode == 'reduce_dual':
                offset_B = len(M.states) + len(M.robust_pairs_suc) + M.robust_constraints
                nr_rows = 2*len(M.states) + len(M.robust_pairs_suc) + M.robust_constraints + M.robust_successors
                
            else:
                offset_B = len(M.states) + len(M.robust_pairs_suc) + 2*M.robust_constraints
                nr_rows = 2*len(M.states) + len(M.robust_pairs_suc) + 2*M.robust_constraints + M.robust_successors
        
            for v, (THETA_SA,THETA) in enumerate(M.parameters.items()):
                for (s_id,a_id) in M.param2stateAction[THETA]:
                    
                    j = M.states_dict[s_id].actions_dict[a_id].alpha_start_idx
                    b = M.states_dict[s_id].actions_dict[a_id].model.b
                    s = M.states_dict[s_id]
                    
                    row[i]  = np.arange(j, j+len(b)) + offset_A
                    col[i]  = np.ones(len(b)) * v
                    data[i] = M.discount * s.policy[a_id] * deriv_valuate(b, THETA) * CVX.cns[s_id].dual_value
        
                    row[i+1]  = np.array([s_id + offset_B])
                    col[i+1]  = np.array([v])
                    data[i+1] = np.array([M.discount * s.policy[a_id] * deriv_valuate(b, THETA) @ CVX.alpha[(s_id, a_id)].value])
                    
                    i += 2
    
        self.Ju = csc_matrix((np.concatenate(data), 
                                 (np.concatenate(row), np.concatenate(col))), 
                                 shape=(nr_rows, len(M.parameters)))
        
        if mode == 'remove_dual':
            self.Ju = self.Ju[-self.J.shape[0]:, :]
        
    def solve_eqsys(self):
        '''
        Compute the Jacobian of the solution in each state with respect to the
        parameter instantiation. The (sparse) Jacobian has |S| rows and |V| 
        columns, with S the set of states and V the set of parameters.

        Returns
        -------
        time : float
            Solver run time.

        '''
    
        print('Compute gradients via linear equation system...')
        
        tocDiff(False)
        gradients = spsolve_umf(self.J, -self.Ju)[:self.num_states, :]
        time = tocDiff(False)
        
        return gradients, time
    
    def solve_inversion(self, CVX):
        
        print('Compute gradients via matrix inversion...')
        
        tocDiff(False)
        
        self.m1 = self.A11
        self.m2 = np.hstack((self.A12[:, CVX.keepalpha], self.A13))
        self.m3 = self.A21
        self.m4 = np.hstack((self.A22[:, CVX.keepalpha].todense(), self.A23.todense()))
        
        self.schur_inv = inv(self.m4 - self.m3 @ self.m2)
        
        MAT = -(self.m1 + self.m2 @ self.schur_inv @ self.m3)
        
        gradients = MAT @ self.Ju[0:self.m1.shape[0], :]
        time = tocDiff(False)
        
        return gradients, time



def solve_cvx(J, Ju, sI, k, solver = 'SCS', verbose = False):
    '''
    Determine 'k' parameters with highest derivative in the initial state

    Parameters
    ----------
    J : Left-hand side matrix
    Ju : Right-hand side matrix
    sI : Stochastic initial state
    k : Number of parameters to select (integer)
    solver : Solver to use (string)
    verbose : If True, give verbose output

    Returns
    -------
    K : indices of chosen parameters
    v : Optimal value of the LP
    time : Time to build and solve LP

    '''
    
    print('Compute parameter importance via LP (CvxPy)...')
    
    tocDiff(False)
    
    cvx = {}
    cvx['x'] = cp.Variable(J.shape[1])
        
    cvx['y'] = cp.Variable(Ju.shape[1], nonneg=True)
    
    cvx['obj'] = cp.Maximize(cvx['x'][ sI['s']] @ sI['p'])
    
    cvx['cns'] = [J @ cvx['x'] == -Ju @ cvx['y'],
                  cp.sum(cvx['y']) == k]
    
    # If more than one parameter is to be selected, constrain y <= 1
    if k > 1:
        cvx['cns'] += [cvx['y'] <= 1]
    
    cvx['prob'] = cp.Problem(objective = cvx['obj'], 
                                  constraints = cvx['cns'])
    
    print('- Call solver {}...'.format(solver))
    
    if solver == 'GUROBI':
        cvx['prob'].solve(solver='GUROBI')
    elif solver == 'SCS':
        cvx['prob'].solve(solver='SCS', requires_grad=True, eps=1e-14, max_iters=20000)
    else:
        cvx['prob'].solve(solver=solver)
    
    if verbose:
        print('Solver status:', cvx['prob'].status)
    
    time = tocDiff(False)
    
    v   = cvx['prob'].value
    y   = cvx['y'].value
    
    # Get the indices of the k parameters showing maximimum derivatives
    K = np.argpartition(y, -k)[-k:]
    
    # # Absolute indices of the k chosen parameters
    # K   = M.param_index[~M.pars_at_max][ind_sort]
    # Y   = y[ind_sort]
    
    return K, v, time

def solve_cvx_gurobi(J, Ju, sI, k, verbose = True):
    '''
    Determine 'k' parameters with highest derivative in the initial state

    Parameters
    ----------
    J : Left-hand side matrix
    Ju : Right-hand side matrix
    sI : Stochastic initial state
    k : Number of parameters to select (integer)

    Returns
    -------
    K : indices of chosen parameters
    v : Optimal value of the LP
    time : Time to build and solve LP

    '''
    
    import gurobipy as gp
    from gurobipy import GRB
    
    print('Compute parameter importance via LP (GurobiPy)...')
    
    tocDiff(False)
    
    m = gp.Model('CVX')
    if verbose:
        m.Params.OutputFlag = 1
    else:
        m.Params.OutputFlag = 0
        
    # m.Params.Method = 1
    m.Params.NumericFocus = 3
    m.Params.ScaleFlag = 2
        
    x = m.addMVar(J.shape[1])
    y = m.addMVar(Ju.shape[1], lb=0, ub=1)
    slack = m.addMVar(J.shape[0])
    
    m.addConstr(J @ x == slack - Ju @ y)
    m.addConstr(gp.quicksum(y) == k)
    
    # m.tune()
    m.setObjective(x[ sI['s']] @ sI['p'] - 1e9*gp.quicksum(slack * slack), GRB.MAXIMIZE)
    
    print('- Call GUROBI')
    m.optimize()
    print('- Maximum slack is: {}'.format(np.max(np.abs(slack.X))))
    
    time = tocDiff(False)
    
    # Get the indices of the k parameters showing maximimum derivatives
    K = np.argpartition(y.X, -k)[-k:]
    
    # # Absolute indices of the k chosen parameters
    # K   = M.param_index[~M.pars_at_max][ind_sort]
    # Y   = y[ind_sort]
    
    return K, m.ObjVal, time



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
                   
                SUM -= M.discount * M.states_dict[s_pre].policy[a_id] * p * self.Vnu_primal[s_pre]
                    
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
                        + M.discount * s.policy[a.id] * valuate(a.model.b) * self.Vnu_primal[s.id] \
                        + valuate(a.model.A) @ self.Vnu_dual[(s.id, a.id)] == - self.Dtheta_Nabla_L[(s.id, a.id)]
                        
                    # 3)
                    # Add a scalar constraint for each robust state-action pair
                    
                    # For each robust state-action pair: the probability of choosing
                    # action a times the Vnu_primal variable, plus the sum of Vnu_dual
                    # over all places where it occurs, is zero.
                    self.Vcns[('g3', s.id, a.id)] = \
                        M.discount * s.policy[a.id] * self.Vnu_primal[s.id] + cp.sum(self.Vnu_dual[(s.id, a.id)]) == 0
                
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
                        
                self.Vcns[('g5a', s.id)] = self.Vx[s.id] + M.discount*SUM == -self.Dtheta_h[s.id]
                    
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
                        Dtheta_h[s.id] += M.discount * s.policy[a.id] * deriv_valuate(a.model.b, theta) @ self.alpha[(s.id, a.id)].value
                    
                        self.Dtheta_Nabla_L[(s.id, a.id)].value = \
                            M.discount * s.policy[a.id] * deriv_valuate(a.model.b, theta) * self.cns[s.id].dual_value
              
        self.Dtheta_h.value = Dtheta_h          
             
        print('\nSolve sensitivity program...')
        
        # Solve optimization problem
        if solver == 'GUROBI':
            self.sens_prob.solve(solver='GUROBI')
        else:
            self.sens_prob.solve(solver='SCS')
            
        print('Status of computing gradients:', self.sens_prob.status)
        
        return self.Vx