# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:22:11 2022

@author: Thom Badings
"""

import numpy as np
from numpy.linalg import inv
from scipy import sparse
from scipy.sparse.linalg import spsolve
# from scikits.umfpack import spsolve as spsolve_umf
from core.commons import tocDiff
from core.commons import valuate, deriv_valuate
import gurobipy as gp
from gurobipy import GRB

class gradient:

    def __init__(self, M, robust_bound, verbose = False):
        '''
        Initialize object to compute gradients for prMCs.

        Parameters
        ----------
        M : prmc object

        Returns
        -------
        None.

        '''
        
        self.robust_bound = robust_bound
        
        A11_row = []
        A11_col = []
        A11_val = []
        
        for s in M.states:
          if not M.is_sink_state(s.id):
            for a in s.actions:
                if not a.robust:                
                    for ss, p_hat in zip(a.successors, a.model.probabilities):
                        
                        A11_row += [s.id]
                        A11_col += [ss]
                        A11_val += [-M.discount * s.policy[int(a.id)] * p_hat]
            
        A11_plus = sparse.csc_matrix((A11_val, (A11_row, A11_col)), 
                                     shape=(len(M.states), len(M.states)))
        self.A11 = sparse.identity(len(M.states)) + A11_plus
        
        # Define A21
        A21_row = []
        A21_col = []
        A21_val = []
        i = 0
        for (s_id,a_id) in M.robust_successors.keys():
            succ = M.states_dict[s_id].actions_dict[a_id].successors
            for ss_id in succ:
                A21_row += [i]
                A21_col += [ss_id]
                if self.robust_bound == 'lower':
                    A21_val += [1]
                else:
                    A21_val += [-1]
                i += 1
        
        robust_successors = sum([a for a in M.robust_successors.values()])
        self.A21 = sparse.csc_matrix((A21_val, (A21_row, A21_col)), 
                                     shape=(robust_successors, len(M.states)))
            
        A12_row = []
        A12_col = []
        A12_val = []
        A13_row = []
        A13_col = []
        A13_val = []
        
        for i, (s_id,a_id) in enumerate(M.robust_successors.keys()):
            s = M.states_dict[s_id]
            
            j = s.actions_dict[a_id].alpha_start_idx
            b = s.actions_dict[a_id].model.b
            
            if self.robust_bound == 'lower':
                A12_vals = M.discount * s.policy[a_id] * valuate(b)
            else:
                A12_vals = -M.discount * s.policy[a_id] * valuate(b)
                
            for l,jj in enumerate(range(j, j+len(b))):
                A12_row += [s_id]
                A12_col += [jj]
                A12_val += [A12_vals[l]]
                
            A13_row += [s_id]
            A13_col += [i]
            if self.robust_bound == 'lower':
                A13_val += [M.discount * s.policy[a_id]]
            else:
                A13_val += [-M.discount * s.policy[a_id]]
        
        self.A12 = sparse.csc_matrix((A12_val, (A12_row, A12_col)), 
                                     shape=( len(M.states), M.robust_constraints ))
        
        self.A13 = sparse.csc_matrix((A13_val, (A13_row, A13_col)), 
                                     shape=( len(M.states), len(M.robust_successors) ))
        
        self.A22 = sparse.block_diag([M.states_dict[s_id].actions_dict[a_id].model.A.T for (s_id,a_id) in M.robust_successors.keys() ], format='csc')
        self.A23 = sparse.block_diag([np.ones(( n, 1 )) for n in M.robust_successors.values() ])
    
    
    def update(self, M, CVX):
        
        # Check if the size of the PRMC agrees with the size of the cvx problem
        assert len(CVX.keepalpha) + len(CVX.keeplambda) == M.robust_constraints
        
        nr_robust_successors = sum([a for a in M.robust_successors.values()])
        
        self.J = sparse.bmat([
                       [ self.A11,  self.A12[:, CVX.keepalpha],   self.A13 ],
                       [ self.A21,  self.A22[:, CVX.keepalpha],   self.A23 ]], format='csc')
        
        nr_rows = len(M.states) + nr_robust_successors
        
        iters = sum([len(a) for a in M.param2stateAction.values()])
        row = [[]]*iters
        col = [[]]*iters
        data = [[]]*iters
        i = 0
        
        for v, THETA_SA in enumerate(M.paramIndex):
            THETA = M.parameters[THETA_SA]
            
            for (s_id,a_id) in M.param2stateAction[THETA_SA]:
                
                b = M.states_dict[s_id].actions_dict[a_id].model.b
                s = M.states_dict[s_id]
                
                row[i]  = np.array([s_id])
                col[i]  = np.array([v])
                if self.robust_bound == 'lower':
                    data[i] = np.array([M.discount * s.policy[a_id] * deriv_valuate(b, THETA) @ CVX.alpha[(s_id, a_id)].X])
                else:
                    data[i] = -np.array([M.discount * s.policy[a_id] * deriv_valuate(b, THETA) @ CVX.alpha[(s_id, a_id)].X])
                
                i += 1
    
        self.Ju = sparse.csc_matrix((np.concatenate(data), 
                                 (np.concatenate(row), np.concatenate(col))), 
                                 shape=(nr_rows, len(M.parameters)))

        
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
       
        
       
def solve_eqsys(J, Ju):
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
    gradients = spsolve(J, -Ju)
    time = tocDiff(False)
    
    return gradients, time



def solve_cvx_gurobi(J, Ju, sI, k, direction = GRB.MAXIMIZE,
                     verbose = True, slackvar = False, method = 2):
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
    Deriv : Array of derivatives

    '''
    
    m = gp.Model('CVX')
    if verbose:
        m.Params.OutputFlag = 1
    else:
        m.Params.OutputFlag = 0
    
    m.Params.Method = method
    m.Params.Seed = 0
    m.Params.Crossover = 0

    # m.Params.SimplexPricing = 3
    # m.Params.NumericFocus = 3
    # m.Params.ScaleFlag = 1
    # m.Params.Presolve = 1
    # m.Params.Crossover = 0
    
    print('--- Define optimization model...')
    
    x = m.addMVar(J.shape[1], lb=-GRB.INFINITY, ub=GRB.INFINITY)
    y = m.addMVar(Ju.shape[1], lb=0, ub=1)
    
    # Switch between adding a (quadratically-penalized) slack variable
    if slackvar:
        slack = m.addMVar(J.shape[0], lb=-0.1, ub=0.1)
        
        # m.tune()
        F = 1e9
        if direction is GRB.MAXIMIZE:
            penalty = -F * (slack @ slack)
        else:
            penalty = F * (slack @ slack)
    
    else:
        slack       = 0
        penalty     = 0
        
    m.addConstr(J @ x == -Ju @ y + slack)
    m.addConstr(gp.quicksum(y) == k)    
    
    m.setObjective(x[ sI['s']] @ sI['p'] + penalty, direction)
    
    print('--- Solve...')
    
    m.optimize()
    
    if slackvar:
        print('- Maximal slack value is {}'.format(np.max(np.abs(slack.X))))
    
    # Get the indices K of the k>=1 parameters showing maximimum derivatives
    K = np.argpartition(y.X, -k)[-k:]
    optimum = m.ObjVal
    
    # If the number of desired parameters >1, then we still need to obtain their values
    if k > 1: 
        
        # If matrix is square, use matrix inversion. Otherwise, use LP.
        if J.shape[0] == J.shape[1]:
            print('- Retrieve actual derivatives for {} parameters via matrix inversion'.format(len(K)))
            Deriv = sparse.linalg.spsolve(J, -Ju[:,K])[sI['s']].T @ sI['p']
        else:
            print('- Retrieve actual derivatives for {} parameters via LP'.format(len(K)))
            Deriv = solve_cvx_single(J, Ju[:,K], sI, direction, method)
            
    else:
        Deriv = np.array([optimum])
    
    return m, K, Deriv


def solve_cvx_single(J, Ju, sI, direction = GRB.MAXIMIZE, method = -1):
    
    m = gp.Model('CVX')
    m.Params.OutputFlag = 1
    
    Deriv = np.zeros(Ju.shape[1])
    
    x = m.addMVar((J.shape[1], Ju.shape[1]), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    
    m.addConstr(J @ x == -Ju)  

    m.setObjective(gp.quicksum(sI['p'] @ x[sI['s'],:]), direction)
    m.optimize()

    Deriv = sI['p'] @ x.X[sI['s'], :] 
    
    return Deriv