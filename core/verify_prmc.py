import numpy as np
import cvxpy as cp
from core.polynomial import polynomial
from copy import copy
import sys
import gurobipy as gp
from gurobipy import GRB

import sympy

class verify_prmc:
    
    def __init__(self, M, R, beta_penalty, robust_bound, verbose = True):
        '''
        Compute the solution for the given prMC using optimization program.

        Parameters
        ----------
        M : prMC object
        R : Reward vector (numpy array)
        beta_penalty : Penalty given to beta (only tested with value of zero)
        robust_bound : Is either 'upper' or 'lower'; indicated whether to get 
            a robust under or over approximation of the solution.
        verbose : Boolean for verbose output

        Returns
        -------
        None.

        '''
        
        self.R = R
        self.robust_bound = robust_bound
        self.verbose = verbose
        
        self.times_solved = 0
        
        if verbose:
            print('Define linear program...')
            print('- Define variables...')
            
        self.cvx = gp.Model('CVX')
            
        self.x = self.cvx.addMVar(len(M.states), lb=0, ub=GRB.INFINITY, name='x')
        self.alpha = {}
        self.beta = {}
        
        for s in M.states: 
            if not M.is_sink_state(s.id):
                for a in s.actions:
                    if a.robust:
                        self.beta[(s.id, a.id)]  = self.cvx.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='beta({},{})'.format(s.id, a.id))
        
        # Objective function
        if beta_penalty == 0:
            penalty = 0
        else:
            penalty = beta_penalty * gp.quicksum([bet for bet in self.beta.values()])
        
        if self.robust_bound == 'lower':
            self.cvx.setObjective(self.x[M.sI['s']] @ M.sI['p'] - penalty, GRB.MAXIMIZE)
        else:
            self.cvx.setObjective(self.x[M.sI['s']] @ M.sI['p'] + penalty, GRB.MINIMIZE)
        
        self.cns = {}
        
        if verbose:
            print('- Define constraints...')
        
        # Constraints per state (under the provided policy)
        for s in M.states:
            
            self.add_state_constraint(s, M)
    
    
    def add_state_constraint(self, s, M):
        '''
        Add constraints for state `s`

        Parameters
        ----------
        s : Stormpy state object (note: not an integer, but the object itself)
        M : prMC object

        Returns
        -------
        None.

        '''
        
        RHS = 0
        
        # If not a terminal state
        if M.is_sink_state(s.id):
            if self.verbose:
                print('-- Add constraint for sink state {}'.format(s.id))
            
            self.cns[s.id] = self.cvx.addConstr(self.x[s.id] == self.R[s.id], name=str(s.id))
                
        else:
            # For each action in the policy
            for a in s.actions:
                
                # If action outcome is robust / uncertain
                if a.robust:
                    
                    self.alpha[(s.id, a.id)] = self.cvx.addMVar(len(a.model.b), lb=0, ub=GRB.INFINITY, name='alpha({},{})'.format(s.id, a.id))
                    
                    if self.verbose:
                        print('-- Add robust constraint for {}'.format(s.id))
                    
                    # Add constraints for an uncertain probability distribution
                    bXalpha = [b.val()*alph if isinstance(b, polynomial) else b*alph
                               for b,alph in zip(a.model.b, self.alpha[(s.id, a.id)])]
                    
                    if self.robust_bound == 'lower':
                    
                        RHS -= s.policy[a.id] * (cp.sum(bXalpha) + self.beta[(s.id, a.id)])    
                    
                        # Add constraints on dual variables for each state-action pair
                        self.cns[(s.id, a.id)] = self.cvx.addConstr(
                            a.model.A.T @ self.alpha[(s.id, a.id)] \
                            + self.x[a.successors] \
                            + self.beta[(s.id, a.id)] == 0,
                            name='({},{})'.format(int(s.id), int(a.id))
                            )
                            
                    else:
                        
                        RHS += s.policy[a.id] * (cp.sum(bXalpha) + self.beta[(s.id, a.id)])    
                        
                        # Add constraints on dual variables for each state-action pair
                        self.cns[(s.id, a.id)] = self.cvx.addConstr(
                            a.model.A.T @ self.alpha[(s.id, a.id)] \
                            - self.x[a.successors] \
                            + self.beta[(s.id, a.id)] == 0,
                            name='({},{})'.format(int(s.id), int(a.id))
                            )
                    
                else:
                    
                    if self.verbose:
                        print('-- Add nonrobust constraint for {}'.format(s.id))
                    
                    # Add constraints for a precise probability distribution
                    RHS += s.policy[a.id] * (self.x[a.model.states] @ a.model.probabilities)
                  
            self.cns[s.id] = self.cvx.addConstr(self.x[s.id] == self.R[s.id] + M.discount * RHS, name=str(s.id))
            
    
    def update_parameter(self, M, var, verbose = False):
        '''
        Update constraints related to a single parameter
        '''
        
        #TODO only remove constraint if it has really changed. Can this be optimized?
        
        s_update = np.unique(np.array(M.param2stateAction[ var ])[:,0])
        
        for s in s_update:
            
            if verbose:
                print('For state #{}'.format(s))
            
            state = M.states_dict[s]
            
            cnstr = self.cvx.getConstrByName(str(s))
            self.cvx.remove(cnstr)
            
            for action in state.actions:
                a = action.id        
                
                if verbose:
                    print('Remove state-action pair ({},{})'.format(s, a))
                
                if (s,a) in self.cns:
                    
                    
                    for name in self.alpha[(s, a)]:
                        self.cvx.remove(name)
                        
                    
                    for n in range(len(action.successors)):
                        
                        name = '({},{})[{}]'.format(int(s), int(a), n)
                        cnstr = self.cvx.getConstrByName(name)                
                        self.cvx.remove(cnstr)
                
            self.add_state_constraint(state, M)
    
    
    def solve(self, verbose = False, store_initial = False):
        '''
        Solve optimization problem.
        '''
        
        if verbose:
            print('Solve linear program problem...')
            self.cvx.Params.OutputFlag = 1
        else:
            self.cvx.Params.OutputFlag = 0
        
        self.cvx.optimize()
        
        # Copy the solution in the initial iteration (for validation purposes)
        if store_initial:
            self.x_tilde = copy(self.x.X)
        
        
    def delta_solve(self, theta, delta, verbose = False):
        '''
        Solve optimization problem with a slight change in one of the params.

        Parameters
        ----------
        theta : TYPE
            DESCRIPTION.
        delta : TYPE
            DESCRIPTION.
        verbose : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        grad_numerical : TYPE
            DESCRIPTION.

        '''
        
        # TODO: function currently not used; can it be useful?
        
        theta.value += delta
        self.cvx.optimize()
        theta.value -= delta
        
        grad_numerical = (self.x.X-self.x_tilde)/delta
        
        return grad_numerical
        
    
    def _add_active_constraint(self, prmc, s, a, alpha_nonzero, cns_needed):
        # Select more constraints to be active
        
        A = prmc.states_dict[s].actions_dict[a].model.A
        basis = A[alpha_nonzero, :]
        
        fixed = False
        
        # Try adding one by one until we have a fully determined point
        for i in range(len(alpha_nonzero)):
            # If constraint i is not yet active
            if not alpha_nonzero[i]:
                # Check if it is independent to current basis
                new_basis = np.concatenate((basis, A[[i], :]))
                
                _, ind_vecs, _ = np.linalg.svd(new_basis)
                
                # If all vectors of new basis are independent,
                # make this constraint active
                if all(ind_vecs != 0):
                    alpha_nonzero[i] = True
                                        
                    basis = A[alpha_nonzero, :]
                    
                    if np.linalg.matrix_rank(basis) == cns_needed:
                        fixed = True
                        break
                    
        if not fixed:
            print('\nError: Could not repair active constraints for state {}'.format(s))
            
        return alpha_nonzero
    
    
    def _remove_active_constraint(self, prmc, s, a, alpha_nonzero, cns_needed):
        
        A = prmc.states_dict[s].actions_dict[a].model.A
        
        basis = A[alpha_nonzero, :]
        basis_idx = np.where(alpha_nonzero)[0]
        
        _, ind_vecs, _ = np.linalg.svd(basis)
        
        # Get linearly independent constraints
        independent_constraints = basis_idx[ (ind_vecs != 0) ]
        
        # Keep minimum number needed of these
        keep_active = independent_constraints[:cns_needed]
        
        alpha_nonzero = np.full_like(alpha_nonzero, False)
        alpha_nonzero[keep_active] = True
        
        return alpha_nonzero
    
    
    def _check_linear_independence(self, prmc, s, a, alpha_nonzero):
        
        A = prmc.states_dict[s].actions_dict[a].model.A
        
        basis = A[alpha_nonzero, :]
        basis_idx = np.where(alpha_nonzero)[0]
        
        _, keep_index = sympy.Matrix(basis.T).rref()
        
        not_index = [i for i in range(len(basis)) if i not in keep_index]
        for i in not_index:
            print('>>> Remove active constraint idx {} due to linear dependence'.format(not_index))
        
        keep_active = basis_idx[list(keep_index)]
        
        alpha_nonzero = np.full_like(alpha_nonzero, False)
        alpha_nonzero[keep_active] = True
        
        return alpha_nonzero
        
    
    def get_active_constraints(self, prmc, verbose = False, repair = False):    
        '''
        Determine which constraints are active, and subsequently determine
        whether complementary slackness is satisfied. Also contains some
        prototype features to repair if the active constraints are under/over-
        specified.

        Parameters
        ----------
        prmc : prMC object
        verbose : Boolean for verbose output
        repair : Boolean; if True, try to repair active constraints if needed

        Returns
        -------
        violated : If True, slackness was violated

        '''
        
        violated = False
    
        self.keeplambda = [[]] * len(self.alpha)
        self.keepalpha = [[]] * len(self.alpha)
        
        self.active_constraints = {}
        
        # Check if assumption is satisfied
        for i,(s,a) in enumerate(self.alpha.keys()):
            
            num_successors = len(prmc.states_dict[s].actions_dict[a].successors)
            cns_needed = num_successors - 1
            
            # Active constraint if alpha is nonzero
            alpha_nonzero = np.abs(self.alpha[(s, a)].X) >= 1e-12

            # lambda_nonzero = np.abs(self.alpha[(s, a)].RC) >= 1e-12
            # both_zero = ~alpha_nonzero + ~lambda_nonzero

            cns_active = sum(alpha_nonzero)
            
            if repair:
                alpha_nonzero = self._check_linear_independence(prmc, s, a, alpha_nonzero)
            
                if not cns_needed == cns_active:
                    
                    # Try to repair by selecting the required number of extra active constraints
                    if cns_needed > cns_active:
                        
                        if verbose:
                            print('\nActivate {} more constraint for state {}...'.format(cns_needed - cns_active, s))
                        alpha_nonzero = self._add_active_constraint(prmc, s, a, alpha_nonzero, cns_needed)
                        
                    elif cns_needed < cns_active:
                        
                        if verbose:
                            print('\nActivate {} less constraint for state {}...'.format(cns_active - cns_needed, s))
                        alpha_nonzero = self._remove_active_constraint(prmc, s, a, alpha_nonzero, cns_needed)                        
                        
                cns_active = sum(alpha_nonzero)
                        
            if not cns_needed == cns_active:
                if verbose:
                    print('\nError: bad number of active constraints encountered for ({},{})'.format(s,a))
                    print('Active constraints:', sum(alpha_nonzero))
                    print('Constraints needed:', cns_needed)
                       
                violated = True
                    
            self.active_constraints[(s, a)] = alpha_nonzero
            
            self.keepalpha[i]  = alpha_nonzero
            self.keeplambda[i] = ~alpha_nonzero
            
        self.keepalpha = np.where( np.concatenate(self.keepalpha) == True )[0]
        self.keeplambda = np.where( np.concatenate(self.keeplambda) == True )[0]
        
        return violated