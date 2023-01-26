# %run "~/documents/sensitivity-prmdps/prmdp-sensitivity-git/generate_slipgrid_pmc.py"

import numpy as np
import json
import stormpy
import os
import math
from pathlib import Path

# Load PRISM model with STORM
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))



def gen_pMC(N, slipping_probabilities, policy_before, policy_after,
                      terrain, model_name, loc_package, loc_warehouse):
    
    '''
    Generates a parametric Markov chain under a fixed policy for the 
    grid world example.
    Exports to a PRISM model file.
    '''

    # Note: first element is y, second element is x
    dct = {-1: (0,  0),
            0:  (0,  1),
            1:  (1,  0),
            2:  (0, -1),
            3:  (-1, 0)
            }
    
    slipping_estimates = {}
    for v,p in slipping_probabilities.items():
        
        slipping_estimates[v] = np.random.binomial(N, p) / N
    
    move_before = np.empty_like(policy_before, dtype=object)
    for x,row in enumerate(policy_before):
        for y,val in enumerate(row):
            move_before[x,y] = dct[val]
            
    move_after = np.empty_like(policy_before, dtype=object)
    for x,row in enumerate(policy_after):
        for y,val in enumerate(row):
            move_after[x,y] = dct[val]
    
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
                Q += ["    [step] (x={}) & (y={}) & (z=1) -> 1: (x'=x) & (y'=y) & (z'=z);".format(x, y)]
            
    Q += ['endmodule\n',
          '// reward structure (number of steps to reach the target)',
          'rewards',
          '    [step] true : 1;',
          'endrewards\n',
          'label "goal" = x={} & y={} & z=1;'.format(loc_warehouse[0], loc_warehouse[1])
          ]
    
    # Print to file
    with open(r'{}.nm'.format(model_name), 'w') as fp:
        fp.write('\n'.join(Q))
        
    with open(r'{}.json'.format(model_name), 'w') as fp:
        json.dump(slipping_probabilities, fp)
        
    with open(r'{}_mle.json'.format(model_name), 'w') as fp:
        json.dump(slipping_estimates, fp)
        
    print('Exported model with name "{}"'.format(model_name))
    
    
    
def gen_pMDP(N, terrain, model_name,
             loc_package, loc_warehouse):
    
    '''
    Generates a parametric MDP for the grid world example of arbitrary size.
    Exports to a PRISM model file.
    '''
    
    # Determine size of the grid
    (Y,X) = terrain.shape  

    # Note: first element is y, second element is x
    dct = {'east':  (0,  1),
           'south':  (1,  0),
           'west':  (0, -1),
           'north':  (-1, 0)
            }
    
    Q = ['mdp\n']
    
    for t in np.unique(terrain):
        Q += ['const double v{};'.format(t)]
        
    Q += ['\n module grid',
          '    x : [0..4] init 0;',
          '    y : [0..4] init 0;',
          '    z : [0..1] init 0;']
    
    for y,row in enumerate(terrain):
        for x,ter in enumerate(row):
            
            # Iterate over possible moves
            for lab, move in dct.items():
                
                # Compute new position
                x_new = x + move[1]
                y_new = y + move[0]
                
                # Check if out of bounds
                if x_new < 0 or x_new >= X:
                    continue
                if y_new < 0 or y_new >= Y:
                    continue
                
                # Check if we have reached the package
                if (x_new, y_new) == loc_package:
                    z_new = 'z+1'
                else:
                    z_new = 'z'
                    
                # Transition for before picking up the package
                Q += ["    [{}] (x={}) & (y={}) & (z=0) -> 1-v{}: (x'={}) & (y'={}) & (z'={}) + v{}: (x'=x) & (y'=y) & (z'=z);".format(lab, x, y,
                                                                                                                                   ter, x_new, y_new, z_new,  
                                                                                                                                   ter)]
       
                # Now add transitions after picking up the package
                Q += ["    [{}] (x={}) & (y={}) & (z=1) -> 1-v{}: (x'={}) & (y'={}) & (z'=z) + v{}: (x'=x) & (y'=y) & (z'=z);".format(lab, x, y,
                                                                                                                                   ter, x_new, y_new,  
                                                                                                                                   ter)]
            
    Q += ['endmodule\n',
          '// reward structure (number of steps to reach the target)',
          'rewards',
          '    [east] true : 1;',
          '    [south] true : 1;',
          '    [west] true : 1;',
          '    [north] true : 1;',
          'endrewards\n',
          'label "goal" = x={} & y={} & z=1;'.format(loc_warehouse[0], loc_warehouse[1])
          ]
    
    # Print to file
    with open(r'{}.nm'.format(model_name), 'w') as fp:
        fp.write('\n'.join(Q))
        
    print('Exported model with name "{}"'.format(model_name))
    
    
    
def gen_pMDP_random(N, terrain, model_name,
                   loc_package, loc_warehouse, reward):
    
    '''
    Generates a parametric DTMC for the grid world example of arbitrary size,
    under a random policy.
    Exports to a PRISM model file.
    '''
    
    # Determine size of the grid
    (Y,X) = terrain.shape  

    # Note: first element is y, second element is x
    dct = {'east':  (0,  1),
           'south':  (1,  0),
           'west':  (0, -1),
           'north':  (-1, 0)
            }
    
    Q = ['dtmc\n']
    
    for t in np.unique(terrain):
        Q += ['const double v{};'.format(t)]
        
    Q += ['\n module grid',
          '    x : [0..{}] init 0;'.format(X-1),
          '    y : [0..{}] init 0;'.format(Y-1),
          '    z : [0..1] init 0;']
    
    for y,row in enumerate(terrain):
        for x,ter in enumerate(row):
            
            rhs_z0 = []
            rhs_z1 = []
            
            # Check how many moves we can make in this state (x,y)
            moves = 0
            if x > 0:
                moves += 1
            if y > 0:
                moves += 1
            if x < X-1:
                moves += 1
            if y < Y-1:
                moves += 1
            
            # Iterate over possible moves
            for lab, move in dct.items():
                
                # Compute new position
                x_new = x + move[1]
                y_new = y + move[0]
                
                # Check if out of bounds
                if x_new < 0 or x_new >= X:
                    continue
                if y_new < 0 or y_new >= Y:
                    continue
                
                # Check if we have reached the package
                if (x_new, y_new) == loc_package:
                    z_new = 'z+1'
                else:
                    z_new = 'z'
                    
                rhs_z0 += ["1/{}*(1-v{}): (x'={}) & (y'={}) & (z'={})".format(int(moves), ter, x_new, y_new, z_new)]
                rhs_z1 += ["1/{}*(1-v{}): (x'={}) & (y'={}) & (z'=z)".format(int(moves), ter, x_new, y_new)]
                    
            rhs_z0 += ["v{}: (x'=x) & (y'=y) & (z'=z)".format(ter)]
            rhs_z1 += ["v{}: (x'=x) & (y'=y) & (z'=z)".format(ter)]
                
            # Transition for before picking up the package
            Q += ["    [step] (x={}) & (y={}) & (z=0) -> {};".format(x, y, " + ".join(rhs_z0))]
            
            # Now add transitions after picking up the package
            Q += ["    [step] (x={}) & (y={}) & (z=1) -> {};".format(x, y, " + ".join(rhs_z1))]
            
            
    Q += ['endmodule\n',
          '// reward structure (number of steps to reach the target)',
          'rewards',
          '    [step] true : {};'.format(reward),
          'endrewards\n',
          'label "goal" = x={} & y={} & z=1;'.format(loc_warehouse[0], loc_warehouse[1])
          ]
    
    prism_path = str(Path(ROOT_DIR, str(model_name)+'.nm'))
    drn_path   = str(Path(ROOT_DIR, str(model_name)+'.drn'))
    
    # Export to PRISM file
    with open(r'{}'.format(prism_path), 'w') as fp:
        fp.write('\n'.join(Q))
        
    print('Exported PRISM model with name "{}"'.format(model_name))
    
    # Convert from PRISM to DRN file
    formula = 'Rmin=? [F \"goal\"]'
    
    program = stormpy.parse_prism_program(prism_path)
    properties = stormpy.parse_properties(formula, program)
    model = stormpy.build_parametric_model(program, properties)
    
    stormpy.export_to_drn(model, drn_path)
    
    return prism_path, json_path, drn_path
    

# %%
##########################################
# Generate motivating example from paper #

np.random.seed(4)

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
N = 10

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

model_name = "models/slipgrid/dummy"

loc_package   = (4, 1)
loc_warehouse = (3, 4)

gen_pMC(N, slipping_probabilities, policy_before, policy_after,
        terrain, model_name, loc_package, loc_warehouse)

# %%

##########################################
# Generate other, random slipgrids

grid_size = [50,100]
no_params = [100, 1000]
p_range = [0.01, 0.02]

# Number of parameters to estimate probabilities with
N = 1000

ITERS = 1

BASH_FILE = ["#!/bin/bash",
             "cd ..;",
             'echo -e "START GRID WORLD EXPERIMENTS...";']

for Z in grid_size:
  for V in no_params:
    for seed in range(ITERS):
                
        np.random.seed(0)
                
        model_name = "models/slipgrid/pmc_size={}_params={}_seed={}".format(Z,V,seed)
        
        # By default, put package in top right corner and warehouse in bottom left
        loc_package   = (Z-2, 1)
        loc_warehouse = (1, Z-2)
        
        if V > Z**2:
            print('Skip configuration, because no. params exceed no. states.')
            continue
        
        # Set ID's of terrain types
        terrain = np.random.randint(low = 0, high = V, size=(Z,Z))
        
        # Set slipping probabilities (v1 corresponds with terrin type 1)
        slipping_probabilities = {
            'v'+str(i): np.random.uniform(p_range[0], p_range[1]) for i in range(V)
            }
        
        # Minimum transition probability estimate
        delta = 1e-4
        
        # Obtain point estimates for each of the transition probabilities
        slipping_estimates = {}
        for v,p in slipping_probabilities.items():
            
            slipping_estimates[v] = max(min(np.random.binomial(N, p) / N, 1-delta), delta)
           
        json_path  = str(Path(ROOT_DIR, str(model_name)+'.json'))
        with open(r'{}.json'.format(model_name), 'w') as fp:
            json.dump(slipping_probabilities, fp)
            
        with open(r'{}_mle.json'.format(model_name), 'w') as fp:
            json.dump(slipping_estimates, fp)
        
        # model_name = "models/slipgrid/input/pmdp_size={}_params={}".format(Z,V)
        # gen_pMDP(N, terrain, model_name,
        #          loc_package, loc_warehouse)
        
        # Scale reward with the size of the grid world
        reward = 10**(-math.floor(math.log(Z**2, 10)))
        
        _, json_path, drn_path = gen_pMDP_random(
            N, terrain, model_name, loc_package, loc_warehouse, reward)
        
        BASH_FILE += ["python3 run_pmc.py --model '{}' --parameters '{}' --formula 'Rmin=? [F \"goal\"]' --verify--validate_delta 0.001 --output_folder 'output/slipgrid/' ".format(drn_path, json_path)]
        
BASH_FILE += ["#", "python3 parse_output.py --folder 'output/slipgrid/' --table_name 'tables/slipgrid_table'"]
        
# Export bash file to perform grid world experiments
with open('experiments/grid_world.sh', 'w') as f:
    f.write("\n".join(BASH_FILE))