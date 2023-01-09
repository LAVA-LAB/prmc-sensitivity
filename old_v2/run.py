import cvxpy as cp
import numpy as np
import copy

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from core.commons import tocDiff
from core.cvx import solve_LP, sensitivity_LP
from models.models import prMDP_3S, prMDP_reza
from core.poly import poly    


M, policy = prMDP_reza()

verbose = True

if verbose:
    print('\nSolve LP for expected reward...')

# Define decision variables
x = cp.Variable(len(M.S), nonneg=True)
alpha = {(s,a): cp.Variable(len(dct['b'])) for (s,a), dct in M.P.items()}
beta  = {(s,a): cp.Variable() for (s,a), dct in M.P.items()}

# Objective function
objective = cp.Maximize( cp.sum([prob*x[s] for s,prob in M.sI.items()]) )

cns = {}

# Constraints per state (under the provided policy)
for s in M.S:
    if verbose:
        print(' - Add constraints for state', s)
        
    # If not a terminal state
    if s not in M.term:
        
        # Add equality constraint on each state reward variable
        cns[s] = x[s] == M.R[s] - cp.sum([ 
                    prob * (cp.sum([ b.expr()*alp if isinstance(b, poly) else b*alp 
                                    for b,alp in zip(M.P[(s,a)]['b'], alpha[(s,a)]) ]) + beta[(s,a)])
                    for a,prob in policy[s].items() ])
        
        # Add constraints on dual variables for each state-action pari
        for a in M.Act:
            cns[(s,a)] = M.P[(s,a)]['A'].T @ alpha[(s,a)] + x[M.P[(s,a)]['sp']] + beta[(s,a)] == 0
        
            cns[('ineq',s,a)] = alpha[(s,a)] >= 0
        
    # If terminal state, reward equals instantaniously given value
    else:
        cns[s] = x[s] == M.R[s]
        
prob = cp.Problem(objective = objective, constraints = cns.values())

if verbose:
    print('Is problem DCP?', prob.is_dcp(dpp=True))

prob.solve(requires_grad=True, eps=1e-14, max_iters=10000, mode='dense')
prob.backward()

if verbose:
    print('Status:', prob.status)
    print('x =',x.value)

######

def e(size, pos):
    v = np.zeros(size)
    v[pos] = 1
    return v

import scipy.linalg as linalg

valuate = np.vectorize(lambda x: x.val() if isinstance(x, poly) else x, 
                     otypes=[float])

deriv_valuate = np.vectorize(lambda x: x.deriv_eval(M.V['alpha']) if isinstance(x, poly) else 0, 
                     otypes=[float])

print('\n\n')

# %%

A_elems = []
dA_elems = []
C_rows = []
D_elems = []
E_cols = []

for s in M.S:
    if any([(s,a) in M.P for a in M.Act]):
        A_elems += [list(policy[s][a] * valuate(M.P[(s,a)]['b'])) for a in M.Act ]
        dA_elems += [list(policy[s][a] * deriv_valuate(M.P[(s,a)]['b'])) for a in M.Act ]
        
        for sp in M.P[(s,a)]['sp']:
            C_rows += [e(len(M.S), sp)]
        D_elems += [valuate(M.P[(s,a)]['A']).T]
        E_cols += [np.ones((len(M.P[(s,a)]['sp']), 1))]
    else:
        A_elems += [[]]
        dA_elems += [[]]

B = np.array([[np.sum([policy[s][a]*beta[(s,a)].value for a in M.Act if (s,a) in M.P]) for s in M.S]]).T

A = np.block([
        [np.eye(len(M.S)),  linalg.block_diag(*A_elems), B],
        [np.array(C_rows),  linalg.block_diag(*D_elems), linalg.block_diag(*E_cols)]
    ])

alphaSum = sum(a.size for a in alpha.values())
G = np.block([
        [np.zeros((alphaSum, len(M.S))), -np.eye(alphaSum), np.zeros((alphaSum, len(beta)))]
    ])

lambda_flat = np.array([cns[('ineq',s,a)].dual_value for a in M.Act for s in M.S if ('ineq',s,a) in cns]).flatten()
nu_flat     = np.concatenate([
        np.array([cns[(s)].dual_value for s in M.S]).flatten(),
        np.array([cns[(s,a)].dual_value for a in M.Act for s in M.S if (s,a) in cns]).flatten()
    ])

decvar_flat = np.concatenate([
                x.value,
                alpha[(0,0)].value,
                [beta[(0,0)].value]
                ])

Dgx = np.block([
        [np.zeros((G.T.shape[0], G.T.shape[0])),    G.T,                                    A.T],
        [np.diag(lambda_flat) @ G,                  np.diag(G @ decvar_flat),               np.zeros((len(lambda_flat), A.T.shape[1]))],
        [A,                                         np.zeros((A.shape[0], G.T.shape[1])),   np.zeros((A.shape[0], A.T.shape[1]))]
    ])

dA = np.block([
        [np.zeros((len(M.S),len(M.S))),     linalg.block_diag(*dA_elems),                   np.zeros(B.shape)],
        [np.zeros(np.array(C_rows).shape),  np.zeros(linalg.block_diag(*D_elems).shape),    np.zeros(linalg.block_diag(*E_cols).shape)]
    ])

Dgv = np.concatenate([
        dA.T @ nu_flat,
        np.zeros(len(lambda_flat)),
        dA @ decvar_flat
    ])

# %%

gradients = np.linalg.solve(Dgx, -Dgv)

print('Gradients for parameter alpha')
print('Analytical:', gradients[0])
print('From CVXPY:', M.V['alpha'].gradient)
print('Dgx condition number:', np.linalg.cond(Dgx))

print('\n\n')

# %%

A_elems = []
dA_elems = []
C_rows = []
D_elems = []
E_cols = []

for s in M.S:
    if any([(s,a) in M.P for a in M.Act]):
        A_elems += [list(policy[s][a] * valuate(M.P[(s,a)]['b'])) for a in M.Act ]
        dA_elems += [list(policy[s][a] * deriv_valuate(M.P[(s,a)]['b'])) for a in M.Act ]
        
        for sp in M.P[(s,a)]['sp']:
            C_rows += [e(len(M.S), sp)]
        D_elems += [valuate(M.P[(s,a)]['A']).T]
        E_cols += [np.ones((len(M.P[(s,a)]['sp']), 1))]
    else:
        A_elems += [[]]
        dA_elems += [[]]

B = np.array([[np.sum([policy[s][a]*beta[(s,a)].value for a in M.Act if (s,a) in M.P]) for s in M.S]]).T

A = np.block([
        [np.eye(len(M.S)),  linalg.block_diag(*A_elems), B],
        [np.array(C_rows),  linalg.block_diag(*D_elems), linalg.block_diag(*E_cols)]
    ])

alphaSum = sum(a.size for a in alpha.values())
G = np.block([
        [np.zeros((alphaSum, len(M.S))), -np.eye(alphaSum), np.zeros((alphaSum, len(beta)))]
    ])

lambda_flat = np.array([cns[('ineq',s,a)].dual_value for a in M.Act for s in M.S if ('ineq',s,a) in cns]).flatten()
nu_flat     = np.concatenate([
        np.array([cns[(s)].dual_value for s in M.S]).flatten(),
        np.array([cns[(s,a)].dual_value for a in M.Act for s in M.S if (s,a) in cns]).flatten()
    ])

decvar_flat = np.concatenate([
                x.value,
                alpha[(0,0)].value,
                [beta[(0,0)].value]
                ])

BLOCK = np.block([
        [np.zeros((len(M.S), A.shape[1]))],
        [np.diag(lambda_flat / np.round(G @ decvar_flat, 6)) @ G],
        [np.zeros((len(beta), A.shape[1]))]
    ])

Dgx = np.block([
        [BLOCK, A.T],
        [A, np.zeros((A.shape[0], A.T.shape[1]))]
    ])

dA = np.block([
        [np.zeros((len(M.S),len(M.S))),     linalg.block_diag(*dA_elems),                   np.zeros(B.shape)],
        [np.zeros(np.array(C_rows).shape),  np.zeros(linalg.block_diag(*D_elems).shape),    np.zeros(linalg.block_diag(*E_cols).shape)]
    ])

Dgv = np.concatenate([
        dA.T @ nu_flat,
        dA @ decvar_flat
    ])

# %%

gradients = np.linalg.solve(Dgx, -Dgv)

print('Gradients for parameter alpha')
print('Analytical:', gradients[0])
print('From CVXPY:', M.V['alpha'].gradient)
print('Dgx condition number:', np.linalg.cond(Dgx))


# params, states, edges, reward, sI = IMC_3state()



# delta = 1e-6

# # Solve initial LP
# tocDiff(False)
# constraints, x, aLow, aUpp = solve_LP(states, sI, edges, states_post, states_pre, states_nonterm, reward)
# tocDiff()

# assert False

# x_orig = copy.deepcopy(x)

# print('Reward in sI:', np.round(x_orig[sI].value, 8))

# # Setup sensitivity LP
# Dth_prob, Dth_x, X, Y, Z = sensitivity_LP(states, edges_fix, states_post, states_pre, states_nonterm,
#                                           constraints, aLow, aUpp)

# # Define for which parameter to differentiate        

# import pandas as pd
# import numpy as np

# results = pd.DataFrame(columns = ['analytical', 'numerical', 'abs.diff.'])

# for key, param in params.items():

#     for s in states:
        
#         X[(s)].value = cp.sum([
#                         - aLow[(s,ss)].value * edges[(s,ss)][0].deriv_eval(param)
#                         + aUpp[(s,ss)].value * edges[(s,ss)][1].deriv_eval(param)        
#                         for ss in states_post[s]])
    
#     for (s,ss),e in edges.items():
        
#         Y[(s,ss)].value = -constraints[('nu',s)].dual_value * edges[(s,ss)][0].deriv_eval(param)
#         Z[(s,ss)].value =  constraints[('nu',s)].dual_value * edges[(s,ss)][1].deriv_eval(param)
                    
#     Dth_prob.solve()
    
    
#     analytical = np.round(Dth_x[sI].value, 6)
    
#     param.value += delta
    
#     _, x_delta, _, _ = solve_LP(states, sI, edges, states_post, states_pre, states_nonterm, reward)
    
#     numerical = np.round((x_delta[sI].value - x_orig[sI].value) / delta, 6)
    
#     param.value -= delta
    
#     diff = np.round(analytical - numerical, 4)
#     results.loc[key] = [analytical, numerical, diff]
    
# print(results)