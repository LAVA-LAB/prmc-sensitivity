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

ns = 5
goal = set({4})
absorbing = set({1})
init = set({0})
sI = 0

trans = {
    0: { 1: [0.27, 0.73], 2: [0.3, 0.7] },
    1: {},
    2: { 3: [0.25, 0.75], 4: [0.3, 0.7] },
    3: { 1: [0.05, 0.15], 4: [0.8, 0.9],  },
    4: {}
    }

d = 1e-4

M = 0

# if M == 0:
#     trans[0][1] += np.array([d, -d])
#     trans[0][2] += np.array([d, -d])
# elif M == 2:
#     trans[2][3] += np.array([d, -d])
#     trans[2][4] += np.array([d, -d])
# elif M == 3:
#     trans[3][1] += np.array([d, -d])
#     trans[3][4] += np.array([d, -d])


if M == 0:
    trans[0][2] += np.array([d, 0])
    
elif M == 2:
    trans[2][4] += np.array([d, 0])

elif M == 3:
    trans[3][1] += np.array([0, -d])

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
    
    constraints += [r[s] <= mu[s] @ g[s],
                    C[s].T @ mu[s] + r == 0]
    
obj = cp.Maximize(r[sI])

prob = cp.Problem(obj, constraints)
prob.solve()

print('\nStatus:', prob.status)
print('Probability:', np.round(r[sI].value, 8), '\n')

for s in range(ns):
    if s in goal or s in absorbing:
        continue
    
    print('Max dual for state',s,'is:', max(mu[s][:-2].value))
    
    for succ,interval in trans[s].items():
        
        slic = 2*succ
        
        print(' -- Max dual for state',s,'to',succ,'is:', np.round(mu[s][slic:slic+2].value, 3))