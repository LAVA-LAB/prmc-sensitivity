import numpy as np
import cvxpy as cp
from core.polynomial import polynomial
from copy import copy
import sys
import gurobipy as gp
from gurobipy import GRB

class cvx_verification_gurobi:
    
    def __init__(self, M, R, robust_bound, verbose = True):
        
        self.times_solved = 0
        
        if verbose:
            print('Define linear program...')
            print('- Define variables...')
            
        self.cvx = gp.Model('CVX')
            
        self.x = self.cvx.addMVar(len(M.states), lb=0, ub=GRB.INFINITY)
        self.alpha = {}
        self.beta = {}
        
        for s in M.states: 
            if not M.is_sink_state(s.id):
                for a in s.actions:
                    if a.robust:
                        self.alpha[(s.id, a.id)] = self.cvx.addMVar(len(a.model.b), lb=0, ub=GRB.INFINITY)
                        self.beta[(s.id, a.id)]  = self.cvx.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
        
        # Objective function
        if M.beta_penalty == 0:
            penalty = 0
        else:
            penalty = M.beta_penalty * gp.quicksum([bet for bet in self.beta.values()])
        
        if robust_bound == 'lower':
            self.cvx.setObjective(self.x[M.sI['s']] @ M.sI['p'] - penalty, GRB.MAXIMIZE)
        else:
            self.cvx.setObjective(self.x[M.sI['s']] @ M.sI['p'] + penalty, GRB.MINIMIZE)
        
        self.cns = {}
        
        if verbose:
            print('- Define constraints...')
        
        # Constraints per state (under the provided policy)
        for s in M.states:
            
            RHS = 0
            
            # If not a terminal state
            if M.is_sink_state(s.id):
                if verbose:
                    print('-- Add constraint for sink state {}'.format(s.id))
                
                self.cns[s.id] = self.cvx.addConstr(self.x[s.id] == 0)
            else:
                # For each action in the policy
                for a in s.actions:
                    
                    # If action outcome is robust / uncertain
                    if a.robust:
                        
                        if verbose:
                            print('-- Add robust constraint for {}'.format(s.id))
                        
                        # Add constraints for an uncertain probability distribution
                        bXalpha = [b.val()*alph if isinstance(b, polynomial) else b*alph
                                   for b,alph in zip(a.model.b, self.alpha[(s.id, a.id)])]
                        
                        if robust_bound == 'lower':
                        
                            RHS -= s.policy[a.id] * (cp.sum(bXalpha) + self.beta[(s.id, a.id)])    
                        
                            # Add constraints on dual variables for each state-action pair
                            self.cns[(s.id, a.id)] = self.cvx.addConstr(
                                a.model.A.T @ self.alpha[(s.id, a.id)] \
                                + self.x[a.successors] \
                                + self.beta[(s.id, a.id)] == 0
                                )
                                
                        else:
                            
                            RHS += s.policy[a.id] * (cp.sum(bXalpha) + self.beta[(s.id, a.id)])    
                            
                            # Add constraints on dual variables for each state-action pair
                            self.cns[(s.id, a.id)] = self.cvx.addConstr(
                                a.model.A.T @ self.alpha[(s.id, a.id)] \
                                - self.x[a.successors] \
                                + self.beta[(s.id, a.id)] == 0
                                )
                    
                        # self.cns[('ineq', s.id, a.id)] = self.cvx.addConstr(self.alpha[(s.id, a.id)] >= 0)
                        
                    else:
                        
                        if verbose:
                            print('-- Add nonrobust constraint for {}'.format(s.id))
                        
                        # Add constraints for a precise probability distribution
                        RHS += s.policy[a.id] * (self.x[a.model.states] @ a.model.probabilities)
                      
                self.cns[s.id] = self.cvx.addConstr(self.x[s.id] == R[s.id] + M.discount * RHS)
    
    
    def solve(self, verbose = False, store_initial = False):
        
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
        
    
    def check_complementary_slackness(self, M):
        
        # Slack of 1 means lambda is nonzero; Slack of -1 means alpha is nonzero
        self.keeplambda = [[]] * len(self.alpha)
        self.keepalpha = [[]] * len(self.alpha)
        
        self.cns_dual = [self.cns[s].Pi for s in range(len(M.states))]
        
        self.alpha_dual = {}
        self.alpha_primal = {}
        
        # Check if assumption is satisfied
        for i,(s,a) in enumerate(self.alpha.keys()):
            
            self.alpha_dual[(s, a)] = self.alpha[(s, a)].RC
            self.alpha_primal[(s, a)] = self.alpha[(s, a)].X
            
            # If both lambda and alpha are zero (anywhere), complementary
            # slackness is violated
            
            lambda_zero = np.abs(self.alpha_dual[(s, a)]) < 1e-12
            # lambda_zero = np.abs(self.cns[('ineq', s, a)].Pi) < 1e-12
            
            alpha_zero  = np.abs(self.alpha_primal[(s, a)]) < 1e-12
            slacksum = np.sum([lambda_zero, alpha_zero], axis=0)
            
            # If the sum is 2 anywhere, than throw an error
            if np.any(slacksum == 2):
                print('\nERROR: lambda[i] > 0 XOR alpha[i] > 0 must be true for each i')
                print('This assumption was not met for state {} and action {}'.format(s,a))
                print('- Lambda is {}'.format(self.alpha_dual[(s, a)]))
                print('- Alpha is {}'.format(self.alpha[(s,a)].X))
                print('- Beta is {}'.format(self.beta[(s,a)].X))
                print('Abort...')
                
                return False
                
            self.keepalpha[i] = lambda_zero
            self.keeplambda[i] = alpha_zero
            
            # active_constraints = sum(self.keepalpha[i])
            # successors = len(M.states_dict[s].actions_dict[a].successors)
            
            # too_many = active_constraints + 1 - successors
            
            
            # if too_many > 0:
            #     print('ERROR! Too many active constraints',too_many)
                
            #     active_idxs = np.where(self.keepalpha[i] == True)[0]
            #     disable = np.array(active_idxs[:too_many], dtype=int)
                
            #     # print(active_idxs, too_many)
            #     # print(disable)
            #     # print(self.keepalpha)
            #     # print(self.keepalpha[i][disable])
                
            #     # Reduce the number of activate constraints manually
            #     # By setting some alpha values to zero..
            #     self.keepalpha[i][disable] = False
                
            # elif too_many < 0:
            #     print('ERROR! Too few active constraints',too_many)
                
            #     inactive_idxs = np.where(self.keepalpha[i] == False)[0]
            #     enable = np.array(inactive_idxs[:np.abs(too_many)], dtype=int)
                
            #     # Reduce the number of activate constraints manually
            #     # By setting some alpha values to zero..
            #     self.keepalpha[i][enable] = True
                
            
        self.keepalpha = np.where( np.concatenate(self.keepalpha) == True )[0]
        self.keeplambda = np.where( np.concatenate(self.keeplambda) == True )[0]
        
        return True