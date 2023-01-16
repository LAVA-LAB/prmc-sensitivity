import numpy as np
import cvxpy as cp
from core.poly import poly
from copy import copy
import sys

class cvx_verification:
    
    def __init__(self, M, verbose = True, pars_as_expressions = True):
        
        self.times_solved = 0
        
        if verbose:
            print('Define linear program...')
            print('- Define variables...')
            
        # Define decision variables
        self.x = cp.Variable(len(M.states), nonneg=True)
        self.alpha = {}
        self.beta  = {}
        
        for s in M.states: 
            if not s.terminal:
                for a in s.actions:
                    if a.robust:
                        self.alpha[(s.id, a.id)] = cp.Variable(len(a.model.b))
                        self.beta[(s.id, a.id)]  = cp.Variable()
        
        # Objective function
        if M.beta_penalty == 0:
            penalty = 0
        else:
            penalty = M.beta_penalty * cp.sum([bet for bet in self.beta.values()])
        objective = cp.Maximize(self.x[M.sI['s']] @ M.sI['p'] - penalty)
        
        self.cns = {}
        r   = {}
        
        if verbose:
            print('- Define constraints...')
        
        # Constraints per state (under the provided policy)
        for s in M.states:
             
            if M.rewards.has_state_rewards:
                r[s] = M.rewards.get_state_reward(s.id)
            elif M.rewards.has_state_action_rewards:
                r[s] = M.rewards.get_state_action_reward(s.id)
            else:
                print('Error: could not parse reward model; abort...')
                sys.exit()
            
            RHS = 0
            
            # If not a terminal state
            if s.terminal:
                self.cns[s.id] = self.x[s.id] == 0
            else:
                # For each action in the policy
                for a in s.actions:
                    
                    # If action outcome is robust / uncertain
                    if a.robust:
                        
                        # Check if parameters should be used as expression or value
                        # Expression is faster when we're validating afterwards
                        # Otherwise value if faster
                        if pars_as_expressions:
                        
                            # Add constraints for an uncertain probability distribution
                            bXalpha = [b.expr()*alph if isinstance(b, poly) else b*alph
                                       for b,alph in zip(a.model.b, self.alpha[(s.id, a.id)])]
                            
                        else:
                            
                            # Add constraints for an uncertain probability distribution
                            bXalpha = [b.val()*alph if isinstance(b, poly) else b*alph
                                       for b,alph in zip(a.model.b, self.alpha[(s.id, a.id)])]
                        
                        RHS -= s.policy[a.id] * (cp.sum(bXalpha) + self.beta[(s.id, a.id)])
                        
                        # Add constraints on dual variables for each state-action pair
                        self.cns[(s.id, a.id)] = \
                            a.model.A.T @ self.alpha[(s.id, a.id)] \
                            + self.x[a.successors] \
                            + self.beta[(s.id, a.id)] == 0
                    
                        self.cns[('ineq', s.id, a.id)] = self.alpha[(s.id, a.id)] >= 0
                        
                    else:
                        
                        # Add constraints for a precise probability distribution
                        RHS += s.policy[a.id] * (self.x[a.model.states] @ a.model.probabilities)
                      
                self.cns[s.id] = self.x[s.id] == r[s] + M.discount * RHS
            
        self.prob = cp.Problem(objective = objective, constraints = self.cns.values())
        
        if verbose:
            if self.prob.is_dcp(dpp=True):
                print('- Program satisfies DCP rule set')
            else:
                print('- Program does not satisfy DCP rule set')
                
            print('')
    
    def solve(self, solver = 'SCS', verbose = False, store_initial = False):
        
        if verbose:
            print('Solve linear program problem...')
        
        if solver == 'GUROBI':
            self.prob.solve(solver='GUROBI', reoptimize=True)
        elif solver == 'SCS':
            self.prob.solve(solver='SCS', requires_grad=True, eps=1e-14, max_iters=20000, mode='dense')
            self.prob.backward()
        else:
            self.prob.solve(solver=solver)
        
        if verbose:
            print('- Status:', self.prob.status)
            
        # Copy the solution in the initial iteration (for validation purposes)
        if store_initial:
            self.x_tilde = copy(self.x.value)
            
        print('')
        
    def delta_solve(self, theta, delta, solver = 'SCS', 
                    verbose = False):
        
        theta.value += delta
        self.solve(solver = solver, verbose = False)
        theta.value -= delta
        
        if self.prob.status != 'optimal':
            print('ERROR: status of cvx problem is "{}"'.format(self.prob.status))
            sys.exit()
        
        grad_numerical = (self.x.value-self.x_tilde)/delta
        
        return grad_numerical
        
    def check_complementary_slackness(self):
        
        print('Check complementary slackness...')
        
        # Slack of 1 means lambda is nonzero; Slack of -1 means alpha is nonzero
        self.keeplambda = [[]] * len(self.alpha)
        self.keepalpha = [[]] * len(self.alpha)
        
        # Check if assumption is satisfied
        for i,(s,a) in enumerate(self.alpha.keys()):
            if ('ineq', s, a) in self.cns:
                
                # If both lambda and alpha are zero (anywhere), complementary
                # slackness is violated
                lambda_zero = np.abs(self.cns[('ineq', s, a)].dual_value) < 1e-12
                alpha_zero  = np.abs(self.alpha[(s, a)].value) < 1e-12
                slacksum = np.sum([lambda_zero, alpha_zero], axis=0)
                
                # If the sum is 2 anywhere, than throw an error
                if np.any(slacksum == 2):
                    print('\nERROR: lambda[i] > 0 XOR alpha[i] > 0 must be true for each i')
                    print('This assumption was not met for state {} and action {}'.format(s,a))
                    print('- Lambda is {}'.format(self.cns[('ineq', s, a)].dual_value))
                    print('- Alpha is {}'.format(self.alpha[(s,a)].value))
                    print('- Beta is {}'.format(self.beta[(s,a)].value))
                    print('Abort...')
                    sys.exit()
                    
                self.keepalpha[i] = lambda_zero
                self.keeplambda[i] = alpha_zero
        
        self.keepalpha = np.where( np.concatenate(self.keepalpha) == True )[0]
        self.keeplambda = np.where( np.concatenate(self.keeplambda) == True )[0]
        
        print('\n')