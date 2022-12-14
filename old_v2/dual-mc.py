# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:54:32 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np

def matrix_trans(ns, transitions):
    # Compute uncertainty set in the form C * p + g >= 0, where p is the 
    # transition probability
    
    C = []
    g = []
    
    for s in range(ns):
        row = np.zeros(ns)
        row[s] = 1
        
        if s in transitions:
            # Add constraints for a state that is a successor state
            interval = transitions[s]
            
            C += [row, -row]
            g += [-interval[1], interval[0]]
            
        else:
            # Add constraint a state that is no successor states
            C += [row, -row]
            g += [0, 0]
        
    row = np.ones(ns)
    
    # Add constraints such the probabilities sum up to one
    C += [row, -row]
    g += [-1, 1]

    return np.array(C), np.array(g)


# Define transition probability intervals of the Markov chain
trans = {
    0: { #0: [cp.Parameter(value=0.15), cp.Parameter(value=0.3)],
         1: [cp.Parameter(value=0.3), cp.Parameter(value=0.9)], 
         2: [cp.Parameter(value=0.2), cp.Parameter(value=0.6)] },
    1: {},
    2: {}
    }

ns = len(trans)

# Define goal / absorbing states
goal = set({1})
absorbing = set({2})

# Initial state
sI = 0

# Define rewards
r = cp.Variable(ns)

constraints = []

for s in goal:
    print('State',s,'is goal')
    constraints += [r[s] == 1]

for s in absorbing:
    print('State',s,'is critical')
    constraints += [r[s] == 0]
    
C = {}
g = {}    

mu = {}

for s in range(ns):
    if s in goal or s in absorbing:
        continue
    
    print('state:', s)
    
    C[s], g[s] = matrix_trans(ns, trans[s])
    
    mu[s] = cp.Variable(len(g[s]), nonneg=True)
    
    constraints += [r[s] <= cp.sum( [i*j for i,j in zip(mu[s], g[s])] ),
                    C[s].T @ mu[s] + r == 0]
    
obj = cp.Maximize(r[sI])

prob = cp.Problem(obj, constraints)
prob.solve(requires_grad=True)
prob.backward()

print('Status:', prob.status)
print('Reward in sI:', r[sI].value)