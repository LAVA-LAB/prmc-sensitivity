# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:54:32 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np
from util import ticDiff, tocDiff

def define_uncertainty_set(intervals):
    '''
    Define the uncertainty set for a certain (interval-values) probability 
    distribution over successor states
    
    Input: dictionary describing the intervals
    Output: matrices A and b such that Ap <= b; list of successor state idxs
    '''
    
    # Number of successor states
    num_succ = len(intervals)

    # Define A matrix
    a = np.array([[1], [-1]])
    A = np.kron(np.eye(num_succ, dtype=int), a)
    
    # Define b vector by enumerating over all successor states
    b = np.concatenate( [[p_upp, -p_low] for [p_low, p_upp] 
                                         in intervals.values()] )
    
    # List of successor state indexes
    succ_states = list(intervals.keys())
    
    return A, b, succ_states


# Define Markov chain
trans = {
    0: {
        0: { 1: [cp.Parameter(value=0.5), cp.Parameter(value=0.9)],
             2: [cp.Parameter(value=0.2), cp.Parameter(value=0.6)]},
        # 1: { 1: [cp.Parameter(value=0.6), cp.Parameter(value=0.9)],
        #       0: [cp.Parameter(value=0.2), cp.Parameter(value=0.5)] }
       },
    1: {},
    2: {}
    #     {
    #     # 0: { 1: [cp.Parameter(value=0.5), cp.Parameter(value=0.7)],
    #     #       3: [cp.Parameter(value=0.2), cp.Parameter(value=0.4)] },
    #     1: { 0: [cp.Parameter(value=0.5), cp.Parameter(value=0.8)],
    #          3: [cp.Parameter(value=0.2), cp.Parameter(value=0.8)] }
    #     },
    # 3: {}
    }

goal = set({1})
absorbing = set({2})
sI = 0

nS = len(trans)

ticDiff()

# Define optimization problem
rew = cp.Variable(nS, nonneg=True)
constraints = []

# Dictionaries to store variables for each state
A = {}
b = {}
lambd = {}
nu = {}
succ_states = {}

print('\nEnumerate over states to build LP...')
for s, actions in trans.items():

    if s in goal:
        print('State',s,'is goal')
        constraints += [rew[s] == 1]
    
    elif s in absorbing:
        print('State',s,'is critical')
        constraints += [rew[s] == 0]
        
    else:
        print('Add constraints for probability intervals in state',s)
        
        # Enumerate over actions
        for a, intervals in actions.items():
        
            # Get uncertainty set
            A[(s,a)], b[(s,a)], succ_states[(s,a)] = define_uncertainty_set(intervals)
            
            # Define dual variables
            lambd[(s,a)] = cp.Variable(len(succ_states[(s,a)])*2, nonneg=True)
            nu[(s,a)] = cp.Variable(1)
            
            # Add constraints for this distribution over successor states
            constraints += [
                    rew[s] == cp.sum([-i * j for i,j in zip(b[(s,a)],lambd[(s,a)])]) - nu[(s,a)],
                    A[(s,a)].T @ lambd[(s,a)] + rew[succ_states[(s,a)]] + nu[(s,a)] == 0
                ]
        
obj = cp.Maximize(rew[sI])
prob = cp.Problem(obj, constraints)

print('\nIs the poblem DCP?', prob.is_dcp(dpp=True))

prob.solve(requires_grad=True, eps=1e-14, max_iters=10000, mode='dense')
prob.backward()

time = tocDiff(False)
print('CVXPY Status:', prob.status)
print('Time to build and solve LP: {:.4f} seconds'.format(time))
print('Reward in sI:', np.round(rew[sI].value, 6))

print('\nSENSITIVITY ANALYSIS')
print(  '--------------------')
for s, actions in trans.items():
    for a, intervals in actions.items():
        for ss, interval in intervals.items():
            print('transition ({:.0f},{:.0f},{:.0f}) gradient: {:.5f} and {:.5f}'.format(s, a, ss, interval[0].gradient, interval[1].gradient))
            
# %%