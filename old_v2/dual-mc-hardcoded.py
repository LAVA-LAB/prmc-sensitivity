# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:54:32 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np

A = np.array([
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1]
    ])

lb01 = cp.Parameter(pos=True)
ub01 = cp.Parameter(pos=True)

lb02 = cp.Parameter(pos=True)
ub02 = cp.Parameter(pos=True)

b = np.array([ub01, -lb01, ub02, -lb02])

rew = cp.Variable(3, nonneg=True)
lambd = cp.Variable(4, nonneg=True)
nu = cp.Variable(1)

constraints = [
        rew[1] == 1,
        rew[2] == 0,
        rew[0] <= cp.sum([-i * j for i,j in zip(b,lambd)]) - nu,
        A.T @ lambd + rew[1:3] + nu == 0
    ]

obj = cp.Maximize(rew[0])
prob = cp.Problem(obj, constraints)

print('Is problem DCP?', prob.is_dcp(dpp=True))

# Define parameter values
lb01.value = 0.30
ub01.value = 0.90
lb02.value = 0.40
ub02.value = 0.60

prob.solve(requires_grad=True, eps=1e-14, max_iters=10000, mode='dense')
prob.backward()

print('Status:', prob.status)
print('Reward in sI:', rew.value)

print('Sensitivity of interval from (s0,s1):', lb01.gradient, ub01.gradient)
print('Sensitivity of interval from (s0,s2):', lb02.gradient, ub02.gradient)