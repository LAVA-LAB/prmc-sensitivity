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
import cvxpy as cp
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
    
    
    def update(self, M, CVX, mode):
        
        nr_robust_successors = sum([a for a in M.robust_successors.values()])
        
        # Check if the size of the PRMC agrees with the size of the cvx problem
        assert len(CVX.keepalpha) + len(CVX.keeplambda) == M.robust_constraints
        
        # Completely remove dual variables lambda and nu
        if mode == 'remove_dual':
            
            self.J = sparse.bmat([
                           [ self.A11,  self.A12[:, CVX.keepalpha],   self.A13 ],
                           [ self.A21,  self.A22[:, CVX.keepalpha],   self.A23 ]], format='csc')
        
        # Reduce by removing redundant alpha and lambda dual variables
        elif mode == 'reduce_dual':
            
            self.Da_f = -sparse.identity(M.robust_constraints, format='csc')
            
            self.J = sparse.bmat([
                           [ None,      None,                         None,     None,                           self.A11.T, self.A21.T ],
                           [ None,      None,                         None,     self.Da_f[:, CVX.keeplambda],   self.A12.T, self.A22.T ],
                           [ None,      None,                         None,     None,                           self.A13.T, self.A23.T ],
                           [ self.A11,  self.A12[:, CVX.keepalpha],   self.A13, None,                           None,       None  ],
                           [ self.A21,  self.A22[:, CVX.keepalpha],   self.A23, None,                           None,       None  ]], format='csc')
    
        # Solve naively for the full system of equations with all dual variables
        else:        
    
            self.Da_f = -sparse.identity(M.robust_constraints, format='csc')        
    
            LaDa_f = sparse.diags(np.concatenate([CVX.alpha[(s, a)].RC for (s,a) in M.robust_successors.keys()]))
            diag_f = sparse.diags(np.concatenate([CVX.alpha[(s, a)].X for (s,a) in M.robust_successors.keys()]))        
    
            self.J = sparse.bmat([
                           [ None,      None,      None,       None,      self.A11.T,   self.A21.T ],
                           [ None,      None,      None,       self.Da_f, self.A12.T,   self.A22.T ],
                           [ None,      None,      None,       None,      self.A13.T,   self.A23.T ],
                           [ None,      LaDa_f,    None,       diag_f,    None,         None  ],
                           [ self.A11,  self.A12,  self.A13,   None,      None,         None  ],
                           [ self.A21,  self.A22,  self.A23,   None,      None,         None  ]], format='csc')
        
        #####
    
        self.col2param = {}
    
        if mode == 'remove_dual':
            
            nr_rows = len(M.states) + nr_robust_successors
            
            iters = sum([len(a) for a in M.param2stateAction.values()])
            row = [[]]*iters
            col = [[]]*iters
            data = [[]]*iters
            i = 0
            
            for v, THETA_SA in enumerate(M.paramIndex):
                THETA = M.parameters[THETA_SA]
                
                self.col2param[v] = THETA_SA
                
                for (s_id,a_id) in M.param2stateAction[THETA_SA]:
                    
                    j = M.states_dict[s_id].actions_dict[a_id].alpha_start_idx
                    b = M.states_dict[s_id].actions_dict[a_id].model.b
                    s = M.states_dict[s_id]
                    
                    row[i]  = np.array([s_id])
                    col[i]  = np.array([v])
                    if self.robust_bound == 'lower':
                        data[i] = np.array([M.discount * s.policy[a_id] * deriv_valuate(b, THETA) @ CVX.alpha[(s_id, a_id)].X])
                    else:
                        data[i] = -np.array([M.discount * s.policy[a_id] * deriv_valuate(b, THETA) @ CVX.alpha[(s_id, a_id)].X])
                    
                    i += 1
            
        else:
            
            iters = sum([len(a) for a in M.param2stateAction.values()])
            row = [[]]*iters
            col = [[]]*iters
            data = [[]]*iters
            i = 0
            
            offset_A = len(M.states)
            
            if mode == 'reduce_dual':
                offset_B = len(M.states) + len(M.robust_successors) + M.robust_constraints
                nr_rows = 2*len(M.states) + len(M.robust_successors) + M.robust_constraints + nr_robust_successors
                
            else:
                offset_B = len(M.states) + len(M.robust_successors) + 2*M.robust_constraints
                nr_rows = 2*len(M.states) + len(M.robust_successors) + 2*M.robust_constraints + nr_robust_successors
        
            for v, THETA_SA in enumerate(M.paramIndex):
                THETA = M.parameters[THETA_SA]
                
                self.col2param[v] = THETA_SA
                
                for (s_id,a_id) in M.param2stateAction[THETA_SA]:
                    
                    j = M.states_dict[s_id].actions_dict[a_id].alpha_start_idx
                    b = M.states_dict[s_id].actions_dict[a_id].model.b
                    s = M.states_dict[s_id]
                    
                    row[i]  = np.arange(j, j+len(b)) + offset_A
                    col[i]  = np.ones(len(b)) * v
                    
                    if self.robust_bound == 'lower':
                        data[i] = M.discount * s.policy[a_id] * deriv_valuate(b, THETA) * CVX.cns_dual[s_id]
                    else:
                        data[i] = -M.discount * s.policy[a_id] * deriv_valuate(b, THETA) * CVX.cns_dual[s_id]
        
                    row[i+1]  = np.array([s_id + offset_B])
                    col[i+1]  = np.array([v])
                    
                    if self.robust_bound == 'lower':
                        data[i+1] = np.array([M.discount * s.policy[a_id] * deriv_valuate(b, THETA) @ CVX.alpha[(s_id, a_id)].X])
                    else:
                        data[i+1] = -np.array([M.discount * s.policy[a_id] * deriv_valuate(b, THETA) @ CVX.alpha[(s_id, a_id)].X])
                    
                    i += 2
    
        self.Ju = sparse.csc_matrix((np.concatenate(data), 
                                 (np.concatenate(row), np.concatenate(col))), 
                                 shape=(nr_rows, len(M.parameters)))
        
        if mode == 'remove_dual':
            self.Ju = self.Ju[-self.J.shape[0]:, :]

        
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



def solve_cvx(J, Ju, sI, k, direction = cp.Maximize, solver = 'SCS', verbose = False):
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
    
    cvx['obj'] = direction(cvx['x'][ sI['s']] @ sI['p'])
    
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
    
    print('')
    
    return K, v, time



def solve_cvx_gurobi(J, Ju, sI, k, direction = GRB.MAXIMIZE,
                     verbose = True, slackvar = False, method = -1):
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
    
    m = gp.Model('CVX')
    if verbose:
        m.Params.OutputFlag = 1
    else:
        m.Params.OutputFlag = 0
    
    m.Params.Method = method
    m.Params.Seed = 0

    # m.Params.SimplexPricing = 3
    m.Params.NumericFocus = 3
    m.Params.ScaleFlag = 1
    # m.Params.Presolve = 1
    # m.Params.Crossover = 0
    
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
    
    # m.tune()
    m.optimize()
    
    if slackvar:
        print('- Maximal slack value is {}'.format(np.max(np.abs(slack.X))))
    
    # Get the indices of the k parameters showing maximimum derivatives
    K = np.argpartition(y.X, -k)[-k:]
    optimum = m.ObjVal
    
    # If the number of desired parameters >1, then we still need to obtain their values
    if k > 1: 
        Deriv = sparse.linalg.spsolve(J, -Ju[:,K])[sI['s']].T @ sI['p']
            
    else:
        Deriv = np.array([optimum])
    
    return K, Deriv