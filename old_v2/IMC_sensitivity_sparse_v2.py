# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:54:32 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np

######

states = set({0,1,2,3})

edges = {
    (0,1): [cp.Parameter(value=0.3), cp.Parameter(value=0.9)],
    (0,2): [cp.Parameter(value=0.2), cp.Parameter(value=0.8)],
    (0,3): [cp.Parameter(value=0.2), cp.Parameter(value=0.3)],
    (2,0): [cp.Parameter(value=0.3), cp.Parameter(value=0.6)],
    (2,3): [cp.Parameter(value=0.35), cp.Parameter(value=0.6)]
    }

states_post = {s: [ss for ss in states if (s,ss) in edges] for s in states}
states_pre  = {s: [ss for ss in states if (ss,s) in edges] for s in states}

# # Define Markov chain
# trans = {
#     0: {
#         1: [cp.Parameter(value=0.3), cp.Parameter(value=0.9)], # (0,1)
#         2: [cp.Parameter(value=0.2), cp.Parameter(value=0.8)], # (0,2)
#         3: [cp.Parameter(value=0.2), cp.Parameter(value=0.3)], # (0,3)
#        },
#     1: {},
#     2: {
#         0: [cp.Parameter(value=0.3), cp.Parameter(value=0.6)], # (2,0)
#         3: [cp.Parameter(value=0.35), cp.Parameter(value=0.6)], # (2,3)
#        },
#     3: {} #{0: [cp.Parameter(value=0.9), cp.Parameter(value=1.0)]}
#     }

# Initial state index
goal = set({1})
absorbing = set({3})
sI = 0

num_states = len(states)
num_trans = len(edges)
num_states_wEdge = num_states - len(goal) - len(absorbing)

states_nonterm = states - goal - absorbing

# %%
# Sparse implementation of reachability computation via LP
sp_x = cp.Variable(num_states, nonneg=True)
sp_aLow = {(s,ss): cp.Variable() for (s,ss) in edges.keys()}
sp_aUpp = {(s,ss): cp.Variable() for (s,ss) in edges.keys()}
sp_beta = {}

reward = np.array([0, 1, 0, 0])

sp_constraints = {}

print('Compute optimal expected reward')
for s in states:
    print(' - Add constraints for state', s)
    
    if s in states_nonterm:
        sp_beta[s] = cp.Variable()
        beta_sub = sp_beta[s]
    else:
        beta_sub = 0

    sp_constraints[('nu',s)] = sp_x[s] == reward[s] + cp.sum([sp_aLow[(s,ss)]*edges[(s,ss)][0] for ss in states_post[s]]) \
                                                    - cp.sum([sp_aUpp[(s,ss)]*edges[(s,ss)][1] for ss in states_post[s]]) \
                                                    - beta_sub
    
for (s,ss),e in edges.items():
    print(' - Add edge from',s,'to',ss)

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

# %%
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
        
diff = {'from': 2, 'to': 3, 'bound': 0}

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