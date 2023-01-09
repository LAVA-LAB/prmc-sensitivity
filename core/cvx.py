import numpy as np
import cvxpy as cp
from core.poly import poly
from copy import copy, deepcopy
import sys
import pandas as pd

from core.commons import unit_vector, valuate, deriv_valuate

class verify_cvx:
    
    def __init__(self, M, verbose = True):
        
        self.times_solved = 0
        
        if verbose:
            print('\nSolve LP for given formula...')
            print('- Define variables')
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
        
        if verbose:
            print('- Define objective')
        
        # Objective function
        if M.PENALTY == 0:
            penalty = 0
        else:
            penalty = M.PENALTY * cp.sum([bet for bet in self.beta.values()])
        objective = cp.Maximize(cp.sum([prob*self.x[s] for s,prob in M.sI.items()]) - penalty)
        
        self.cns = {}
        r   = {}
        
        self.RHS = {}
        
        # Constraints per state (under the provided policy)
        for s in M.states:
            if verbose:
                print('- Add constraints for state', s.id)
             
            if M.rewards.has_state_rewards:
                r[s] = M.rewards.get_state_reward(s.id)
            elif M.rewards.has_state_action_rewards:
                r[s] = M.rewards.get_state_action_reward(s.id)
            else:
                print('Error: could not parse reward model; abort...')
                sys.exit()
            
            self.RHS[s.id] = 0
            
            # If not a terminal state
            if s.terminal:
                self.cns[s.id] = self.x[s.id] == 0
            else:
                # For each action in the policy
                for a in s.actions:
                    
                    # If action outcome is robust / uncertain
                    if a.robust:
                        
                        if verbose:
                            print('-- Action {} in state {} is robust'.format(a.id, s.id))
                        
                        # Add constraints for an uncertain probability distribution
                        bXalpha = [b.expr()*alph if isinstance(b, poly) else b*alph
                                   for b,alph in zip(a.model.b, self.alpha[(s.id, a.id)])]
                        
                        self.RHS[s.id] -= s.policy[a.id] * (cp.sum(bXalpha) + self.beta[(s.id, a.id)])
                        
                        # Add constraints on dual variables for each state-action pair
                        self.cns[(s.id, a.id)] = \
                            a.model.A.T @ self.alpha[(s.id, a.id)] \
                            + self.x[a.successors] \
                            + self.beta[(s.id, a.id)] == 0
                    
                        self.cns[('ineq', s.id, a.id)] = self.alpha[(s.id, a.id)] >= 0
                        
                    else:
                        if verbose:
                            print('-- Action {} in state {} is precise'.format(a.id, s.id))
                        
                        # Add constraints for a precise probability distribution
                        self.RHS[s.id] += s.policy[a.id] * (self.x[a.model.states] @ a.model.probabilities)
                      
                self.cns[s.id] = self.x[s.id] == r[s] + M.DISCOUNT * self.RHS[s.id]
            
        self.prob = cp.Problem(objective = objective, constraints = self.cns.values())
        
        if verbose:
            if self.prob.is_dcp(dpp=True):
                print('\nProgram satisfies DCP rule set')
            else:
                print('\nProgram does not satisfy DCP rule set')
    
    def solve(self, solver = 'SCS', verbose = False, store_initial = False):
        
        print('Solve optimization problem...')
        
        if solver == 'GUROBI':
            self.prob.solve(solver='GUROBI', reoptimize=True)
        elif solver == 'SCS':
            self.prob.solve(solver='SCS', requires_grad=True, eps=1e-14, max_iters=20000, mode='dense')
            self.prob.backward()
        else:
            self.prob.solve(solver=solver)
        
        if verbose:
            print('Status:', self.prob.status)
            
        # Copy the solution in the initial iteration (for validation purposes)
        if store_initial:
            self.x_tilde = copy(self.x.value)
        
    def delta_solve(self, theta, delta, grad_analytical, solver = 'SCS', 
                    verbose = False):
        
        theta.value += delta
        self.solve(solver = solver, verbose = True)
        theta.value -= delta
        
        grad_numerical = (self.x.value-self.x_tilde)/delta
        grad_diff = np.round(grad_analytical - grad_numerical, 6)
        cum_diff = sum(grad_diff)
        
        if verbose:
            dct = {'analytical': grad_analytical,
                    'experimental': grad_numerical,
                    'abs.diff.': grad_diff}
            
            results = pd.DataFrame(dct)
            print('Cumulative difference:', cum_diff)
            print(results)
        
        return cum_diff
        
    def check_complementary_slackness(self):
        
        print('\nCheck complementary slackness...')
        
        # Check if assumption is satisfied
        for (s,a) in self.alpha.keys():
            if ('ineq', s, a) in self.cns:
                lambda_zero = np.abs(self.cns[('ineq', s, a)].dual_value) < 1e-9
                alpha_zero  = np.abs(self.alpha[(s, a)].value) < 1e-9
                both_zero   = np.sum([lambda_zero, alpha_zero], axis=0)
                
                if not np.all(both_zero == 1):
                    print('\nERROR: lambda[i] > 0 XOR alpha[i] > 0 must be true for each i')
                    print('This assumption was not met for state {} and action {}'.format(s,a))
                    print('- Lambda is {}'.format(self.cns[('ineq', s, a)].dual_value))
                    print('- Alpha is {}'.format(self.alpha[(s,a)].value))
                    print('- Beta is {}'.format(self.beta[(s,a)].value))
                    print('Abort...')
                    sys.exit()
                    
                