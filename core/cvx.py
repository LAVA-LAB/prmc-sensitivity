import numpy as np
import cvxpy as cp
from core.poly import poly
from copy import copy
import sys
import pandas as pd

class verify_cvx:
    
    def __init__(self, M, PI, verbose = True):
        
        self.times_solved = 0
        
        DISCOUNT = 1.0
        ALPHA_PENALTY = 1e-7
        
        if verbose:
            print('\nSolve LP for given formula...')
        
        # Define decision variables
        self.x = cp.Variable(len(M.states), nonneg=True)
        self.alpha = {}
        self.beta  = {}
        
        for s in M.states: 
            if not s.terminal:
                for a,prob in PI[s.id].items():                
                    if a.robust:
                        self.alpha[(s.id, a.id)] = cp.Variable(len(a.model.b))
                        self.beta[(s.id, a.id)]  = cp.Variable()
        
        # Objective function
        penalty = ALPHA_PENALTY * cp.sum([ cp.sum(alph) for alph in self.alpha.values()])
        objective = cp.Maximize(cp.sum([prob*self.x[s] for s,prob in M.sI.items()]) - penalty)
        
        self.cns = {}
        RHS = {}
        r   = {}
        
        # Constraints per state (under the provided policy)
        for s in M.states:
            if verbose:
                print('- Add constraints for state', s.id)
             
            if M.rewards.has_state_rewards:
                r[s] = M.rewards.get_state_reward(s.id) + s.id*0.01 
            elif M.rewards.has_state_action_rewards:
                r[s] = M.rewards.get_state_action_reward(s.id) + s.id*0.01
            else:
                print('Error: could not parse reward model; abort...')
                sys.exit()
            
            RHS = 0
            
            # If not a terminal state
            if not s.terminal:
                # For each action in the policy
                for a,prob in PI[s.id].items():
                    
                    # If action outcome is robust / uncertain
                    if a.robust:
                        
                        if verbose:
                            print('-- Action {} in state {} is robust'.format(s.id, a.id))
                        
                        # Add constraints for an uncertain probability distribution
                        bXalpha = [b.expr()*alph if isinstance(b, poly) else b*alph
                                   for b,alph in zip(a.model.b, self.alpha[(s.id, a.id)])]
                        
                        RHS -= prob * (cp.sum(bXalpha) + self.beta[(s.id, a.id)])
                        
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
                        RHS += prob * (self.x[a.model.states] @ a.model.probabilities)
                      
            self.cns[s.id] = self.x[s.id] <= r[s] + DISCOUNT * RHS
                
        self.prob = cp.Problem(objective = objective, constraints = self.cns.values())
        
        if verbose:
            if self.prob.is_dcp(dpp=True):
                print('Program satisfies DCP rule set')
            else:
                print('Program does not satisfy DCP rule set')
    
    def solve(self, solver = 'SCS', verbose = False):
        
        if solver == 'GUROBI':
            self.prob.solve(solver='GUROBI')
        else:
            self.prob.solve(solver='SCS', requires_grad=True, eps=1e-14, max_iters=10000, mode='dense')
            self.prob.backward()
        
        if verbose:
            print('Status:', self.prob.status)
            
        # Copy the solution in the initial iteration (for validation purposes)
        if self.times_solved == 0:
            self.x_base = copy(self.x.value)
        self.times_solved += 1
        
    def delta_solve(self, theta, delta, grad_analytical, solver = 'SCS', 
                    verbose = False):
        
        theta.value += delta
        self.solve(solver = solver)
        theta.value -= delta
        
        grad_numerical = (self.x.value-self.x_base)/delta
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
        
        print('Check complementary slackness...')
        
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
                    print('Abort...')
                    sys.exit()