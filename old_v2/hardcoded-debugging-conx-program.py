# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:55:56 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np

h_x = {0: cp.Variable(),
       1: cp.Variable(),
       2: cp.Variable(),
       3: cp.Variable(),
       4: cp.Variable(),}
h_r = np.array([1, 2, 3, 4, 5])
h_alpha = {
    0: cp.Variable(4),
    2: cp.Variable(4)
    }
h_beta = {
    0: cp.Variable(),
    2: cp.Variable()
    }

v0 = cp.Parameter(value = 0.1)
v2 = cp.Parameter(value = 0.1)

A0 = np.array([
    [-1, -1],
    [-1,  1],
    [1,  -1],
    [1,   1]
    ])
A2 = np.array([
    [-1, -1],
    [-1,  1],
    [1,  -1],
    [1,   1]
    ])

c = [
      h_x[0] == h_r[0] - (  (-1.0 + v0) * h_alpha[0][0] + v0 * h_alpha[0][1] + v0 * h_alpha[0][2] + (1+v0) * h_alpha[0][3] + h_beta[0] ),
      h_x[1] == h_r[1],
      h_x[2] == h_r[2] - (  (-1.0 + v2) * h_alpha[2][0] + v2 * h_alpha[2][1] + v2 * h_alpha[2][2] + (1+v2) * h_alpha[2][3] + h_beta[2] ),
      h_x[3] == h_r[3],
      h_x[4] == h_r[4],
      A0.T[0,:] @ h_alpha[0] + h_x[1] + h_beta[0] == 0,
      A0.T[1,:] @ h_alpha[0] + h_x[2] + h_beta[0] == 0,
      A2.T[0,:] @ h_alpha[2] + h_x[3] + h_beta[2] == 0,
      A2.T[1,:] @ h_alpha[2] + h_x[4] + h_beta[2] == 0,
     ]
c_ineq = [
      h_alpha[0] >= 0,
      h_alpha[2] >= 0
      ]

G = np.block([np.zeros((8, 5)), -np.eye(8), np.zeros((8, 2))])
A1 = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    ])
A2 = np.array([
    [(-1.0 + v0.value), v0.value, v0.value, (1+v0.value), 0, 0, 0 ,0],
    np.zeros(8),
    [0, 0, 0, 0, (-1.0 + v2.value), v2.value, v2.value, (1+v2.value)],
    np.zeros(8),
    np.zeros(8),
    [-1, -1, 1, 1, 0, 0, 0, 0],
    [-1, 1, -1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, -1, 1, 1],
    [0, 0, 0, 0, -1, 1, -1, 1],
    ]) 
dA2 = np.array([
    [0, 0, 0, 0, 0, 0, 0 ,0],
    np.zeros(8),
    [0, 0, 0, 0, 1, 1, 1, 1],
    np.zeros(8),
    np.zeros(8),
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    ]) 
A3 = np.array([
    [1, 0],
    np.zeros(2),
    [0, 1],
    np.zeros(2),
    np.zeros(2),
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1]
    ])
A = np.block([A1, A2, A3])
dA = np.block([np.zeros_like(A1), dA2, np.zeros_like(A3)])

h_prob = cp.Problem(objective = cp.Maximize(h_x[0]), constraints=c + c_ineq)

solver = 'SCS'

if solver == 'SCS':
    h_prob.solve('SCS', requires_grad=True)
    h_prob.backward()
else:
    h_prob.solve('GUROBI')
print('Hardcoded solution:\nValue:',h_x[0].value,h_x[2].value,'\nGradient:',v0.gradient,v2.gradient)

# print('Lambdas:', c[7].dual_value, c[8].dual_value)
# print('Alphas:', h_alpha[0].value, h_alpha[2].value)


x_vec = np.array([h_x[i].value for i in range(5)])
alpha_vec = np.concatenate([ h_alpha[0].value, h_alpha[2].value ])
beta_vec  = np.array([ h_beta[0].value, h_beta[2].value ])

xab_vec = np.concatenate([x_vec, alpha_vec, beta_vec])

lambda_vec = np.concatenate([ c_ineq[0].dual_value, c_ineq[1].dual_value ])
nu_vec = np.array([c[i].dual_value for i in range(len(c))])

h_Dxg = np.block([
    [np.zeros((15,15)), G.T, A.T],
    [np.diag(lambda_vec) @ G, np.diag(G @ xab_vec), np.zeros((8, 9))],
    [A, np.zeros((9, 8)), np.zeros((9, 9))]
    ])

h_Dvg = np.concatenate([
    dA.T @ nu_vec,
    np.zeros(8),
    dA @ xab_vec
    ])

sol = np.linalg.solve(h_Dxg, -h_Dvg)
print('SOL:', sol)

import copy

delta = 0.00001 
x_old = copy.copy(h_x[0].value)

# v0.value += delta
# 
# h_prob.solve('GUROBI')
# print('Experimental gradient V0:', (h_x[0].value - x_old) / delta)

# v0.value -= delta
v2.value += delta

h_prob.solve(solver)
print('\nExperimental gradient V2:', (h_x[0].value - x_old) / delta)

# assert False