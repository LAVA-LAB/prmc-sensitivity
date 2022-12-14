import cvxpy as cp
import numpy as np
import copy

from commons import tocDiff
from cvx import solve_LP, sensitivity_LP
from models import IMC1, IMC_3state
    
params, states, edges, reward, sI = IMC_3state()

edges_fix = copy.deepcopy(edges)

states_post = {s: [ss for ss in states if (s,ss) in edges] for s in states}
states_pre  = {s: [ss for ss in states if (ss,s) in edges] for s in states}

terminal = set([s for s in states if len(states_post[s]) == 0])
states_nonterm = states - terminal

delta = 1e-6

# Solve initial LP
tocDiff(False)
constraints, x, aLow, aUpp = solve_LP(states, sI, edges, states_post, states_pre, states_nonterm, reward)
tocDiff()

assert False

x_orig = copy.deepcopy(x)

print('Reward in sI:', np.round(x_orig[sI].value, 8))

# Setup sensitivity LP
Dth_prob, Dth_x, X, Y, Z = sensitivity_LP(states, edges_fix, states_post, states_pre, states_nonterm,
                                          constraints, aLow, aUpp)

# Define for which parameter to differentiate        

import pandas as pd
import numpy as np

results = pd.DataFrame(columns = ['analytical', 'numerical', 'abs.diff.'])

for key, param in params.items():

    for s in states:
        
        X[(s)].value = cp.sum([
                        - aLow[(s,ss)].value * edges[(s,ss)][0].deriv_eval(param)
                        + aUpp[(s,ss)].value * edges[(s,ss)][1].deriv_eval(param)        
                        for ss in states_post[s]])
    
    for (s,ss),e in edges.items():
        
        Y[(s,ss)].value = -constraints[('nu',s)].dual_value * edges[(s,ss)][0].deriv_eval(param)
        Z[(s,ss)].value =  constraints[('nu',s)].dual_value * edges[(s,ss)][1].deriv_eval(param)
                    
    Dth_prob.solve()
    
    
    analytical = np.round(Dth_x[sI].value, 6)
    
    param.value += delta
    
    _, x_delta, _, _ = solve_LP(states, sI, edges, states_post, states_pre, states_nonterm, reward)
    
    numerical = np.round((x_delta[sI].value - x_orig[sI].value) / delta, 6)
    
    param.value -= delta
    
    diff = np.round(analytical - numerical, 4)
    results.loc[key] = [analytical, numerical, diff]
    
print(results)