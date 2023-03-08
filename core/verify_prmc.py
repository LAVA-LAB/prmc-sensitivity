import numpy as np
import cvxpy as cp
from core.polynomial import polynomial
from copy import copy
import sys
import gurobipy as gp
from gurobipy import GRB

import sympy

class verify_prmc:
    
    def __init__(self, M, R, robust_bound, verbose = True):
        
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
        if M.beta_penalty == 0:
            penalty = 0
        else:
            penalty = M.beta_penalty * gp.quicksum([bet for bet in self.beta.values()])
        
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
                
                    # self.cns[('ineq', s.id, a.id)] = self.cvx.addConstr(self.alpha[(s.id, a.id)] >= 0)
                    
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
        
        self.cvx.write('out.lp')
        # assert False
        
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
        
        # print(A)
        # print('\n\n Basis is', basis)
        
        # Try adding one by one until we have a fully determined point
        for i in range(len(alpha_nonzero)):
            # If constraint i is not yet active
            if not alpha_nonzero[i]:
                # Check if it is independent to current basis
                new_basis = np.concatenate((basis, A[[i], :]))
                
                # print('Try adding {}'.format(i))
                # print('New basis:')
                # print(new_basis)
                
                _, ind_vecs, _ = np.linalg.svd(new_basis)
                
                # print(ind_vecs)
                
                # If all vectors of new basis are independent,
                # make this constraint active
                if all(ind_vecs != 0):
                    alpha_nonzero[i] = True
                                        
                    basis = A[alpha_nonzero, :]
                    
                    # print('- Make constraint {} active'.format(i))
                    
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
        
        # print('- Keep constraints {} active'.format(keep_active))
    
        return alpha_nonzero
    
    
    def _check_linear_independence(self, prmc, s, a, alpha_nonzero):
        
        # print('Check state {}'.format(s))
        
        A = prmc.states_dict[s].actions_dict[a].model.A
        
        basis = A[alpha_nonzero, :]
        basis_idx = np.where(alpha_nonzero)[0]
        
        # print('Basis idx:', basis_idx)
        # print(basis)
        
        echelon, indep_index = sympy.Matrix(basis.T).rref()
        
        not_index = [i for i in range(len(basis)) if i not in indep_index]
        if len(not_index) > 0:
            print('>>> Remove active constraint idx {} due to linear independence'.format(not_index))
        
        # print('- Linearly dependent active constraints detected (idx: {})'.format(not_index))
        
        # print(indep_index)
        
        keep_active = basis_idx[list(indep_index)]
        
        alpha_nonzero = np.full_like(alpha_nonzero, False)
        alpha_nonzero[keep_active] = True
        
        
        return alpha_nonzero
        
    
    def get_active_constraints(self, prmc, verbose = False, repair = False):    
    
        violated = False
    
        self.keeplambda = [[]] * len(self.alpha)
        self.keepalpha = [[]] * len(self.alpha)
    
        self.cns_dual = [self.cns[s].Pi for s in range(len(prmc.states))]
        
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
            
            if not cns_needed == cns_active and repair:
                    
                print('Repair!')
                
                # Try to repair by selecting the required number of extra active constraints
                if cns_needed > cns_active:
                    
                    # print('\nActivate {} more constraint for state {}...'.format(cns_needed - cns_active, s))
                    alpha_nonzero = self._add_active_constraint(prmc, s, a, alpha_nonzero, cns_needed)
                    
                elif cns_needed < cns_active:
                    
                    # print('\nActivate {} less constraint for state {}...'.format(cns_active - cns_needed, s))
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


                    
    
    '''
    def check_complementary_slackness(self, M, verbose = False):
        
        from core.commons import valuate, deriv_valuate
        
        violated = False
        repair = True
        
        # Slack of 1 means lambda is nonzero; Slack of -1 means alpha is nonzero
        self.keeplambda = [[]] * len(self.alpha)
        self.keepalpha = [[]] * len(self.alpha)
        
        self.cns_dual = [self.cns[s].Pi for s in range(len(M.states))]
        
        self.alpha_dual = {}
        self.alpha_primal = {}
        
        self.active_constraints = {}
        
        
        
        # Check if assumption is satisfied
        for i,(s,a) in enumerate(self.alpha.keys()):
            
            self.alpha_dual[(s, a)] = self.alpha[(s, a)].RC
            self.alpha_primal[(s, a)] = self.alpha[(s, a)].X
            
            # If both lambda and alpha are zero (anywhere), complementary
            # slackness is violated
            
            lambda_zero = np.abs(self.alpha_dual[(s, a)]) < 1e-12
            # lambda_zero = np.abs(self.cns[('ineq', s, a)].Pi) < 1e-12
            
            alpha_zero  = np.abs(self.alpha_primal[(s, a)]) < 1e-12
            alpha_nonzero  = np.abs(self.alpha_primal[(s, a)]) >= 1e-12
            slacksum = np.sum([lambda_zero, alpha_zero], axis=0)
            
            self.active_constraints[(s, a)] = ~alpha_zero
            
            # If the sum is 2 anywhere, than throw an error
            if np.any(slacksum == 2):
                if verbose:
                    print('state {}, action {}'.format(s, a))
                    
                    print('\nERROR: lambda[i] > 0 XOR alpha[i] > 0 must be true for each i')
                    print('This assumption was not met for state {} and action {}'.format(s,a))
                    # print('- Lambda is {}'.format(self.alpha_dual[(s, a)]))
                    # print('- Alpha is {}'.format(self.alpha[(s,a)].X))
                    # print('- Beta is {}'.format(self.beta[(s,a)].X))
                    
                    # print('A matrix:', M.states_dict[s].actions_dict[a].model.A)
                    # print('b vector:', valuate(M.states_dict[s].actions_dict[a].model.b))
                    
                    print('Number of active constraints (nonzero alpha): {}'.format(sum(alpha_nonzero)))
                    print('Number of active constraints (zero lambda): {}'.format(sum(lambda_zero)))
                    print('Successor states: {}'.format(len(M.states_dict[s].actions_dict[a].successors)))
                
                violated = True
                
            self.keepalpha[i] = lambda_zero #~alpha_zero
            self.keeplambda[i] = alpha_zero
            
        self.keepalpha = np.where( np.concatenate(self.keepalpha) == True )[0]
        self.keeplambda = np.where( np.concatenate(self.keeplambda) == True )[0]
        
        return violated
    '''    
    
    def tie_break_activate_constraints(self):
        
        return