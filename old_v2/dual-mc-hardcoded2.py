# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:54:32 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np

lb01 = cp.Parameter(pos=True)
ub01 = cp.Parameter(pos=True)

lb02 = cp.Parameter(pos=True)
ub02 = cp.Parameter(pos=True)

dim_r = 3
dim_lambd = 4
dim_nu = 1
dim_x = dim_r + dim_lambd + dim_nu

Q = np.zeros((dim_x,dim_x))
q = np.block([np.zeros(dim_r), np.zeros(dim_lambd), np.zeros(dim_nu)])
q[0] = -1

G = np.block([np.zeros((dim_lambd,dim_r)), -np.eye((dim_lambd)), np.zeros((dim_lambd,dim_nu))])
h = np.zeros(dim_lambd)

A_sub = np.array([
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1]
    ])
b_sub = np.array([-lb01, ub01, -lb02, ub02])

A = np.block([
          [1, np.zeros(dim_r-1), b_sub.T, 1],
          [0, 1, 0, np.zeros(5)],
          [0, 0, 1, np.zeros(5)],
          [np.array([[0,1,0],[0,0,1]]), A_sub.T, np.ones((2,1))]
        ])

b = np.array([0, 1, 0, 0, 0])

x = cp.Variable(dim_x)

constraints = [
    G @ x <= h,
    cp.hstack([cp.sum([i*j for i,j in zip(A_row, x)]) for A_row in A]) == b
    # A @ x == b
    ]

obj = cp.Minimize(q @ x)
prob = cp.Problem(obj, constraints)

print('Is problem DCP?', prob.is_dcp(dpp=True))

# Define parameter values
lb01.value = 0.30
ub01.value = 0.90
lb02.value = 0.40
ub02.value = 0.65

b_sub_value = np.array([b.value for b in b_sub])
A_value = np.block([
          [1, np.zeros(dim_r-1), b_sub_value.T, 1],
          [0, 1, 0, np.zeros(5)],
          [0, 0, 1, np.zeros(5)],
          [np.array([[0,1,0],[0,0,1]]), A_sub.T, np.ones((2,1))]
        ])

prob.solve(requires_grad=True, eps=1e-14, max_iters=10000, mode='dense')
prob.backward()

print('Status:', prob.status)
print('Reward in sI:', np.round(x.value, 5))

####

lambd = constraints[0].dual_value
nu    = constraints[1].dual_value

Dx_g = np.block([
            [Q,                     G.T,                              A_value.T],
            [np.diag(lambd) @ G,    np.diag(G @ x.value - h),         np.zeros((len(lambd), A.shape[0]))],
            [A_value,               np.zeros((A.shape[0], len(h))),    np.zeros((A.shape[0], A.shape[0]))]
        ])

Dx_g_inv = np.linalg.inv(Dx_g)
print('\nMatrix Dx(g) is invertible')

alt_vec = np.diag(np.tile([-1,1], 2))

Dth_g = np.block([
            [np.zeros((dim_r, dim_lambd))],
            [np.diag([-nu[0], nu[0], -nu[0], nu[0]])],
            [np.zeros((dim_nu, dim_lambd))],
            #
            [np.zeros((dim_lambd, dim_lambd))],
            #
            [alt_vec @ x.value[dim_r:dim_r + dim_lambd]],
            [np.zeros((len(nu)-1, dim_lambd))]
        ])

Dth_s = - Dx_g_inv @ Dth_g

print('Analytical gradients:', np.round(Dth_s[0,:], 5))

cp_grad = np.array([
        lb01.gradient, ub01.gradient, lb02.gradient, ub02.gradient
    ])
print('CVXPY gradients:', np.round(cp_grad, 5))

# %%

z = cp.Variable(Dth_s.shape)

aux_prob = cp.Problem(objective = cp.Minimize(0), constraints = [Dx_g @ z == -Dth_g])

aux_prob.solve()
print('Optimal z:', np.round(z.value[0,:], 5))