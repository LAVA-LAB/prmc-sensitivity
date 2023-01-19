# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:35:22 2023

@author: Thom Badings
"""

import numpy as np
import json

np.random.seed(4)

#####################
# Empty starts here #

# Set ID's of terrain types
terrain = np.array([
    [1, 1, 3, 3, 3],
    [2, 1, 3, 3, 4],
    [2, 1, 4, 4, 4],
    [2, 2, 2, 5, 5],
    [2, 2, 2, 5, 5]
    ])

# Set slipping probabilities (v1 corresponds with terrin type 1)
slipping_probabilities = {
    'v1': 0.1,
    'v2': 0.5,
    'v3': 0.4,
    'v4': 0.2,
    'v5': 0.3
    }

# Generate rough estimate of the slipping probabilities
slipping_estimates = {}
N = 20
for v,p in slipping_probabilities.items():
    
    slipping_estimates[v] = np.random.binomial(N, p) / N

# 0 = right, 1 = down, 2 = left, 3 = up
# Policy before the package has been picked up
policy_before = np.array([
    [0, 1, 1, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 0, 3],
    [0, 3, 3, 3, 3],
    [0, 3, 3, 3, 3]
    ])

# Policy after the package has been picked up
policy_after = np.array([
    [0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, -1, 2]
    ])

# Note: first element is y, second element is x
dct = {-1: (0,  0),
       0:  (0,  1),
       1:  (1,  0),
       2:  (0, -1),
       3:  (-1, 0)
       }

# Input ends here   #
#####################

move_before = np.empty_like(policy_before, dtype=object)
for x,row in enumerate(policy_before):
    for y,val in enumerate(row):
        move_before[x,y] = dct[val]
        
move_after = np.empty_like(policy_before, dtype=object)
for x,row in enumerate(policy_after):
    for y,val in enumerate(row):
        move_after[x,y] = dct[val]

loc_package   = (4, 1)
loc_warehouse = (3, 4)

Q = ['dtmc\n']

for t in np.unique(terrain):
    Q += ['const double v{};'.format(t)]
    
Q += ['\n module grid',
      '    x : [0..4] init 0;',
      '    y : [0..4] init 0;',
      '    z : [0..1] init 0;']

for y,row in enumerate(move_before):
    for x,val in enumerate(row):
        
        ter = terrain[y,x]
        
        if (x+val[1], y+val[0]) == loc_package:
            z_new = 'z+1'
        else:
            z_new = 'z'
            
        if val == (0, 1):
            Q += ["    [step] (x={}) & (y={}) & (z=0) -> 1-v{}: (x'=x+1) & (y'=y) & (z'={}) + v{}: (x'=x) & (y'=y) & (z'=z);".format(x, y, ter, z_new, ter)]
        elif val == (0, -1):
            Q += ["    [step] (x={}) & (y={}) & (z=0) -> 1-v{}: (x'=x-1) & (y'=y) & (z'={}) + v{}: (x'=x) & (y'=y) & (z'=z);".format(x, y, ter, z_new, ter)]
        elif val == (1, 0):
            Q += ["    [step] (x={}) & (y={}) & (z=0) -> 1-v{}: (x'=x) & (y'=y+1) & (z'={}) + v{}: (x'=x) & (y'=y) & (z'=z);".format(x, y, ter, z_new, ter)]
        elif val == (-1, 0):
            Q += ["    [step] (x={}) & (y={}) & (z=0) -> 1-v{}: (x'=x) & (y'=y-1) & (z'={}) + v{}: (x'=x) & (y'=y) & (z'=z);".format(x, y, ter, z_new, ter)]
        elif val == (0, 0):
            Q += ["    [step] (x={}) & (y={}) & (z=0) -> 1: (x'=x) & (y'=y) & (z'=z);".format(x, y)]
            
for y,row in enumerate(move_after):
    for x,val in enumerate(row):
        
        ter = terrain[y,x]
        
        if val == (0, 1):
            Q += ["    [step] (x={}) & (y={}) & (z=1) -> 1-v{}: (x'=x+1) & (y'=y) & (z'=z) + v{}: (x'=x) & (y'=y) & (z'=z);".format(x, y, ter, ter)]
        elif val == (0, -1):
            Q += ["    [step] (x={}) & (y={}) & (z=1) -> 1-v{}: (x'=x-1) & (y'=y) & (z'=z) + v{}: (x'=x) & (y'=y) & (z'=z);".format(x, y, ter, ter)]
        elif val == (1, 0):
            Q += ["    [step] (x={}) & (y={}) & (z=1) -> 1-v{}: (x'=x) & (y'=y+1) & (z'=z) + v{}: (x'=x) & (y'=y) & (z'=z);".format(x, y, ter, ter)]
        elif val == (-1, 0):
            Q += ["    [step] (x={}) & (y={}) & (z=1) -> 1-v{}: (x'=x) & (y'=y-1) & (z'=z) + v{}: (x'=x) & (y'=y) & (z'=z);".format(x, y, ter, ter)]
        elif val == (0, 0):
            Q += ["    [step] (x={}) & (y={}) & (z=0) -> 1: (x'=x) & (y'=y) & (z'=z);".format(x, y)]
        
Q += ['endmodule\n',
      '// reward structure (number of steps to reach the target)',
      'rewards',
      '    [step] true : 1;',
      'endrewards\n',
      'label "goal" = x={} & y={} & z=1;'.format(loc_warehouse[0], loc_warehouse[1])
      ]

model_name = "slipgrid_fixed"

# Print to file
with open(r'{}.nm'.format(model_name), 'w') as fp:
    fp.write('\n'.join(Q))
    
with open(r'{}.json'.format(model_name), 'w') as fp:
    json.dump(slipping_probabilities, fp)
    
with open(r'{}_mle.json'.format(model_name), 'w') as fp:
    json.dump(slipping_estimates, fp)
