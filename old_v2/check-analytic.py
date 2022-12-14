# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 20:07:50 2022

@author: Thom Badings
"""

import numpy as np
import cvxpy as cp

Q = np.zeros((8,8))
q = np.block([-1, 0, 0, np.zeros(4), 0])

G = np.diag([-1, -1, -1, -1, -1, -1, -1, 0])[:-1,:]
h = np.zeros(7)

b_sub = np.array([[0.9], [-0.5], [0.6], [-0.2]])
A_sub = np.array([[1, 0],
               [-1, 0],
               [0, 1],
               [0, -1]])

A = np.block([[1, 0, 0, b_sub.T, 1],
              [0, 1, 0, np.zeros((1,4)), 0],
              [0, 0, 1, np.zeros((1,4)), 0],
              [np.zeros((2,1)), np.array([[1], [0]]), np.array([[0], [1]]), A_sub.T, np.ones((2,1))]])
b = np.array([0, 1, 0, 0, 0]) #np.array([[0], [1], [0], [0], [0]])

lambd = np.random.rand(7)
x     = np.random.rand(8)

Dxg = np.block([[Q,                    G.T,        A.T],
                [np.diag(lambd) @ G,   np.diag(G @ x - h),     np.zeros((7,5))],
                [A,                    np.zeros((5, 7)),       np.zeros((5,5))]])

Dxg_inv = np.linalg.inv(Dxg)

#####

x = cp.Variable(len(q))

constraints = [
    G @ x <= h,
    A @ x == b
    ]

obj = cp.Minimize(q @ x)
prob = cp.Problem(obj, constraints)

print('\nIs the poblem DCP?', prob.is_dcp(dpp=True))

prob.solve(requires_grad=True, eps=1e-14, max_iters=10000, mode='dense')
prob.backward()

print('CVXPY Status:', prob.status)
print('Reward in sI:', np.round(x[0].value, 6))

####

lambd = constraints[0].dual_value
nu    = constraints[1].dual_value

Dtheta_g = np.block([
                [np.zeros((3,4))],
                [np.diag([nu[0], -nu[0], nu[0], -nu[0]])],
                [np.zeros((1,4))],
                [np.zeros((7,4))],
                [np.array([x[3].value, -x[4].value, x[5].value, -x[6].value])],
                [np.zeros((4,4))]
                ])

####

# Compute final Jacobian
Dtheta_s = - Dxg_inv @ Dtheta_g


