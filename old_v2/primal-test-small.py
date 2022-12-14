# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:54:32 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np

def matrix_trans(ns, transitions):
    
    C = []
    g = []
    
    for s in range(ns):
        row = np.zeros(ns)
        row[s] = 1
        
        if s in transitions:
            interval = transitions[s]
            
            C += [row, -row]
            g += [-interval[1], interval[0]]
            
        else:
            C += [row, -row]
            g += [0, 0]
        
    row = np.ones(ns)
    
    C += [row, -row]
    g += [-1, 1]

    return np.array(C), np.array(g)



trans = {
    0: { 1: cp.Parameter(2, value=[0.5, 0.9]), 
         2: cp.Parameter(2, value=[0.2, 0.4]) },
    1: {},
    2: {}
    }

ns = len(trans)

goal = set({1})
absorbing = set({2})
sI = 0

# Define rewards
r = cp.Variable(ns)
p = {} # Probabilities within their intervals

constraints = []

for s in goal:
    print('State',s,'is goal')
    constraints += [r[s] == 1]

for s in absorbing:
    print('State',s,'is critical')
    constraints += [r[s] == 0]
    
for s in range(ns):
    if s in goal or s in absorbing:
        continue
    
    print('state:', s)
    
    # Probabilities used in state s
    p[s] = cp.Variable(ns, nonneg=True)
    
    # Sum of probabilities must be one
    constraints += [cp.sum(p[s]) == 1]
    
    constraints += [r[s] == cp.sum(p[s] * r)]
    
    for s_prime in range(ns):
        if s_prime in trans[s]:
            
            
        else:
            constraints += [p[s][s_prime] == 0]
    
    
    for i, (s_prime, intv) in enumerate(trans[s].items()):
        print(intv.value)
        
        constraints += [intv[0] <= p[s][i], p[s][i] <= intv[1]]
    
obj = cp.Maximize(r[sI])

prob = cp.Problem(obj, constraints)
prob.solve()
# prob.backward()

print('Status:', prob.status)
print('Reward in sI:', r[sI].value)