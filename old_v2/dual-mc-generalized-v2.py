# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:54:32 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np
from util import ticDiff, tocDiff

def basis_vec(size, index):
    if type(index) == int:
        arr = np.zeros(size, dtype=int)
        arr[index] = 1
    
    else:
        arr = np.zeros((len(index), size), dtype=int)
        for r,i in enumerate(index):
            arr[r,i] = 1
        
    return arr

def define_uncertainty_set(intervals, dim_lambd):
    '''
    Define the uncertainty set for a certain (interval-values) probability 
    distribution over successor states
    
    Input: dictionary describing the intervals
    Output: matrices A and b such that Ap <= b; list of successor state idxs
    '''
    
    # Number of successor states
    num_succ = len(intervals)

    # Define A matrix
    a = np.array([[-1], [1]])
    
    A = np.zeros((dim_lambd, num_succ), dtype=int)
    b = np.zeros(dim_lambd, dtype=object)
    b_value = np.zeros((dim_lambd))
    
    for j, [p_low, p_upp, trans_id] in enumerate(intervals.values()):
        
        i = 2*trans_id
        
        A[i:i+2, [j]] = np.array([[-1], [1]])
        
        b[i:i+2]    = np.array([-p_low, p_upp])
        b_value[i:i+2] = np.array([-p_low.value, p_upp.value])
    
    # List of successor state indexes
    succ_states = list(intervals.keys())
    
    return A, b, b_value, succ_states

######

# Define Markov chain
trans = {
    0: {
        1: [cp.Parameter(value=0.3), cp.Parameter(value=0.9), 0], # (0,1)
        2: [cp.Parameter(value=0.2), cp.Parameter(value=0.8), 1], # (0,2)
        3: [cp.Parameter(value=0.2), cp.Parameter(value=0.3), 2], # (0,3)
       },
    1: {},
    2: {
        0: [cp.Parameter(value=0.3), cp.Parameter(value=0.6), 3], # (2,0)
        3: [cp.Parameter(value=0.4), cp.Parameter(value=0.6), 4], # (2,3)
       },
    3: {}
    }

# Initial state index
goal = set({1})
absorbing = set({3})
sI = 0

num_states = len(trans)
num_trans = sum([len(tr) for tr in trans.values()])
num_states_wEdge = num_states - len(goal) - len(absorbing)

######

# Size of decision variables of primal problem (with inner constraints dualized)
dim_lambd = 2*num_trans                     # Lambda dual variable of inner problem
dim_nu = num_states_wEdge                   # Nu dual variable of inner problem
dim_x = num_states + dim_lambd + dim_nu     # State vector

######

# Define matrices of objective function
Q = np.zeros((dim_x,dim_x))
q = np.block([np.zeros(num_states), np.zeros(dim_lambd), np.zeros(dim_nu)])
q[sI] = -1

# Define matrices G x <= h of inequality constraints
G = np.block([np.zeros((dim_lambd,num_states)), -np.eye((dim_lambd)), np.zeros((dim_lambd,dim_nu))])
h = np.zeros(dim_lambd)

# Define matrices [A B].T x == [b c].T of equality constraints
A1 = []
A1_value = []
b1 = []
A2 = []
b2 = []

param2state = np.zeros(dim_lambd, dtype=int)
trans2param = {}

polytope_idx   = 0
transition_sum = 0

# Enumerate over all states
print('\nEnumerate over states to build LP...')

for s,intervals in trans.items():
    # If state is a goal state
    if s in goal:
        row = np.block([
                basis_vec(num_states, s), np.zeros(dim_lambd), np.zeros(dim_nu)
                ])
        
        A1 += [row]
        A1_value += [row]
        b1 += [1]
        
    # If state is an absorbing state
    elif s in absorbing:
        row = np.block([
                basis_vec(num_states, s), np.zeros(dim_lambd), np.zeros(dim_nu)
                ])
        
        A1 += [row]
        A1_value += [row]        
        b1 += [0]
    
    # Otherwise, add constraints related to transitions
    else:
        
        # Obtain constraints on uncertainty set in the form of Ax-b <= 0
        poly_A, poly_b, poly_b_value, succ_states = define_uncertainty_set(intervals, dim_lambd)
        
        # Add to upper half of A matrix
        A1 += [np.block([
                basis_vec(num_states, s), poly_b.T, basis_vec(dim_nu, polytope_idx)
                ])]
        
        A1_value += [np.block([
                basis_vec(num_states, s), poly_b_value.T, basis_vec(dim_nu, polytope_idx)
                ])]

        b1 += [0]
        
        # Store mapping from parameters to outgoing state
        param2state[2*transition_sum : 2*(transition_sum + len(succ_states))] = s
        
        # Store mapping from transitions to parameters
        for succ_s in succ_states:
            trans2param[(s, succ_s)] = np.array([2*transition_sum, 
                                                 2*transition_sum+1])
            transition_sum += 1
        
        ####
        
        # Add to lower half of A matrix
        basis_matrix = np.zeros((len(succ_states), dim_nu))
        basis_matrix[:, polytope_idx] = 1

        A2 += [np.block([
                basis_vec(num_states, succ_states), poly_A.T, basis_matrix
                ])]
        
        b2 += [0] * len(succ_states)
        
        ####
        
        polytope_idx   += 1
        
# Concatenate A and b to obtain full matrix and vector
A           = np.concatenate(( np.vstack(A1), np.vstack(A2) ))
A_value     = np.concatenate(( np.vstack(A1_value), np.vstack(A2) ))
b           = np.array(b1 + b2)

# %%

from copy import deepcopy

# Define Linear program
x = cp.Variable(dim_x)

# TODO: A @ x == b does not work because of parameters in A..
constraints = [
    G @ x <= h,
    cp.hstack([cp.sum([i*j for i,j in zip(A_row, x)]) for A_row in A]) == b
    ]

obj = cp.Minimize(q @ x)
prob = cp.Problem(obj, constraints)

print('Is problem DCP?', prob.is_dcp(dpp=True))

prob.solve(requires_grad=True, eps=1e-14, max_iters=10000, mode='dense')
prob.backward()

print('Status:', prob.status)
print('Reward in sI:', np.round(x.value, 5))

####

lambd = constraints[0].dual_value
nu    = constraints[1].dual_value

lambd_inner = x[num_states : num_states+dim_lambd].value
beta_inner  = x[num_states + dim_lambd :].value

Dx_g = np.block([
            [Q,                     G.T,                              A_value.T],
            [np.diag(lambd) @ G,    np.diag(G @ x.value - h),         np.zeros((len(lambd), A.shape[0]))],
            [A_value,               np.zeros((A.shape[0], len(h))),   np.zeros((A.shape[0], A.shape[0]))]
        ])

Dx_g_inv = np.linalg.inv(Dx_g)
print('\nMatrix Dx(g) is invertible')

# Alternating [-1, 1] diagonal matrix
alt_vec = np.diag(np.tile([-1,1], num_trans))

# Create slice of the A matrix
A_partial = deepcopy( A_value[0:num_states, num_states:num_states+dim_lambd] )
A_partial[A_partial > 0] = 1
A_partial[A_partial < 0] = -1

# Compute dAx
dAx = A_partial * lambd_inner

Dth_g = np.block([
            [np.zeros((num_states, dim_lambd))],
            [np.diag(alt_vec @ nu[param2state])],
            [np.zeros((dim_nu, dim_lambd))],
            #
            [np.zeros((dim_lambd, dim_lambd))],
            #
            [dAx],
            [np.zeros((num_trans, dim_lambd))]
        ])

Dth_s = - Dx_g_inv @ Dth_g
gradients = Dth_s[0,:]

print('Gradients:')
for tr,params in trans2param.items():
    print(' -- Transition', tr, 'gradients: {:.5f} and {:.5f}'.format(gradients[params[0]], gradients[params[1]]))
    
####
# %%

# Compute gradient via optimization

Glambda = cp.Variable((len(lambd), dim_lambd))
Gnu     = cp.Variable((len(nu), dim_lambd))
Galpha  = cp.Variable((len(lambd_inner), dim_lambd))
Gbeta   = cp.Variable((len(beta_inner), dim_lambd))
Greward = cp.Variable((num_states, dim_lambd))

Gconstraints = [
    # A_value.T @ Gnu == -np.eye(dim_x)[:,num_states:-dim_nu] @ Glambda + np.eye(dim_x)[:,num_states:-dim_nu] @ np.diag(alt_vec @ nu[param2state]),
    # np.diag(-lambd) @ Galpha == np.diag(lambd_inner) @ Glambda,
    
    - np.diag(np.concatenate((np.ones(num_states), lambd_inner, np.ones(dim_nu)))) @ A_value.T @ Gnu == 
        np.eye(dim_x)[:,num_states:-dim_nu] @ np.diag(lambd) @ Galpha 
        - np.eye(dim_x)[:,num_states:-dim_nu] @ np.diag(lambd_inner * (alt_vec @ nu[param2state])),
    A_value @ cp.vstack((Greward, Galpha, Gbeta)) == np.vstack((dAx, np.zeros((num_trans, dim_lambd))))
    ]

Gprob   = cp.Problem(objective = cp.Maximize(cp.sum(Greward)),
                     constraints = Gconstraints)

Gprob.solve()

rew1 = deepcopy(Greward.value)

print('\nGradients in initial state via LP:')
print(np.round(Greward.value, 5))

# %%
from numpy.linalg import inv

A1 = Q
B1 = np.block([G.T, A_value.T])
C1 = np.block([[ np.diag(lambd) @ G ], [A_value]])
D1 = np.block([[np.diag(G @ x.value - h), np.zeros((len(lambd), A.shape[0]))],
              [np.zeros((A.shape[0], len(h))),    np.eye(A.shape[0])] ])

A2 = -inv(B1 @ inv(D1) @ C1)
B2 = inv(B1 @ inv(D1) @ C1) @ B1 @ inv(D1)
C2 = inv(D1) @ C1 @ inv(B1 @ inv(D1) @ C1)
D2 = inv(D1) - inv(D1) @ C1 @ inv(B1 @ inv(D1) @ C1) @ B1 @ inv(D1)

Z = np.block([[ A2, B2 ], [C2, D2]])

Z_nu = deepcopy(Z)
Z_nu[:, 0:-9] = 0

xyz = ( inv(np.eye(35) - Z_nu) @ Z @ -(Dth_g)  )

print('\nGradients, analytically computed:')
print(np.round(xyz[0:num_states, :], 5))

# %%

# Sparse implementation

sp_x = cp.Variable(num_states, nonneg=True)
sp_aLow = {s: {} for s in range(num_states)}
sp_aUpp = {s: {} for s in range(num_states)}
sp_beta = {}

reward = np.array([0, 1, 0, 0])

sp_constraints = {}

print('Compute optimal expected reward')
for s,intervals in trans.items():
    print(' - Add constraints for state', s)
    
    if len(intervals) > 0:
        sp_beta[s] = cp.Variable()
        beta_sub = sp_beta[s]
    else:
        beta_sub = 0
    
    for ss in intervals.keys():
        sp_aLow[s][ss] = cp.Variable()
        sp_aUpp[s][ss] = cp.Variable()
        
        sp_constraints[('la_low',s,ss)] = sp_aLow[s][ss] >= 0 
        sp_constraints[('la_upp',s,ss)] = sp_aUpp[s][ss] >= 0
        
        sp_constraints[('mu',s,ss)] = sp_aUpp[s][ss] - sp_aLow[s][ss] + sp_x[ss] + sp_beta[s] == 0
        
    sp_constraints[('nu',s)] = sp_x[s] == reward[s] + cp.sum([sp_aLow[s][ss]*intv[0] for ss,intv in intervals.items()]) \
                                            - cp.sum([sp_aUpp[s][ss]*intv[1] for ss,intv in intervals.items()]) \
                                            - beta_sub
    
# Concatenate all constraints into a list
sp_prob = cp.Problem(objective = cp.Maximize(sp_x[sI]), constraints = sp_constraints.values())
print('Is problem DCP?', sp_prob.is_dcp(dpp=True))

sp_prob.solve(requires_grad=True, eps=1e-14, max_iters=10000, mode='dense')
sp_prob.backward()

print('Status:', sp_prob.status)
print('Reward in sI:', np.round(sp_x[sI].value, 5))

###
# Perform sensitivity analysis

Dth_x = cp.Variable(num_states)
Dth_aLow = {s: {} for s in range(num_states)}
Dth_aUpp = {s: {} for s in range(num_states)}
Dth_beta = {}

Dth_nu = cp.Variable(num_states)
Dth_mu = {s: {} for s in range(num_states)}
Dth_la_low = {s: {} for s in range(num_states)}
Dth_la_upp = {s: {} for s in range(num_states)}

# Define decision variables
for s,intervals in trans.items():
    if len(intervals) > 0:
        Dth_beta[s] = cp.Variable()
    
    for ss in intervals.keys():
        Dth_aLow[s][ss] = cp.Variable()
        Dth_aUpp[s][ss] = cp.Variable()
        
        Dth_la_low[s][ss] = cp.Variable()
        Dth_la_upp[s][ss] = cp.Variable()
        
        Dth_mu[s][ss] = cp.Variable()

GC = []

# Differentiate lower bound from 0 to 1
X = {}
Y = {}
Z = {}

print('Compute Jacobian of decision variables with respect to parameters')
# Enumerate over states
for s,intervals in trans.items():
    print(' - Add constraints for state', s)
    
    if len(intervals) > 0:
        Dth_beta[s] = cp.Variable()
        beta_sub = Dth_beta[s]
    else:
        beta_sub = 0
    
    # 1
    GC += [Dth_nu[s] + cp.sum([Dth_mu[ss][s] if s in trans[ss] else 0 for ss in range(num_states)]) == 0]
    
    # 3
    if len(intervals) > 0:
        GC += [Dth_nu[s] + cp.sum([Dth_mu[s][ss] for ss in intervals.keys()]) == 0]
       
    # 5
    X[(s)] = cp.Parameter()
    GC += [Dth_x[s] + cp.sum([-intv[0]*Dth_aLow[s][ss] + intv[1]*Dth_aUpp[s][ss] for ss, intv in intervals.items()]) + beta_sub == -X[(s)]]
       
    # Enumerate over edges
    for ss, intv in intervals.items():
        Y[(s,ss)] = cp.Parameter()
        Z[(s,ss)] = cp.Parameter()
        
        GC += [ 
            # 2
            -Dth_la_low[s][ss] - intv[0]*Dth_nu[s] - Dth_mu[s][ss] == Y[(s,ss)],
            -Dth_la_upp[s][ss] + intv[1]*Dth_nu[s] + Dth_mu[s][ss] == -Z[(s,ss)],
            # 4
            sp_constraints[('la_low',s,ss)].dual_value * Dth_aLow[s][ss] == sp_aLow[s][ss].value * Dth_la_low[s][ss],
            sp_constraints[('la_upp',s,ss)].dual_value * Dth_aUpp[s][ss] == sp_aUpp[s][ss].value * Dth_la_upp[s][ss],
            # 6
            Dth_x[ss] - Dth_aLow[s][ss] + Dth_aUpp[s][ss] + Dth_beta[s] == 0
            ]
        
diff = {'from': 0, 'to': 1, 'bound': 0}

for s,intervals in trans.items():
    if s != diff['from']:
        X[(s)].value = 0
    else:
        if diff['bound'] == 0:
            X[(s)].value = -sp_aLow[s][diff['to']].value
        else:
            X[(s)].value = sp_aUpp[s][diff['to']].value
        
    for ss, intv in intervals.items():
        if s != diff['from'] or ss != diff['to']:
            Y[(s,ss)].value = 0
            Z[(s,ss)].value = 0
        else:
            if diff['bound'] == 0:
                Y[(s,ss)].value = -sp_constraints[('nu',s)].dual_value
                Z[(s,ss)].value = 0
            else:
                Y[(s,ss)].value = 0
                Z[(s,ss)].value = sp_constraints[('nu',s)].dual_value
                
Dth_prob = cp.Problem(cp.Maximize(0), constraints = GC)
Dth_prob.solve()

print('Status:', Dth_prob.status)
print('Reward in sI:', np.round(Dth_x[sI].value, 5))