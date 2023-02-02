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
    
    print('')
    
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
        
        # Export tuple of MLE and sample size N
        slipping_estimates[v] = [np.random.binomial(N[v], p) / N[v], N[v]]
    
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
    with open(r'{}.nm'.format(Path(ROOT_DIR,model_name)), 'w') as fp:
        fp.write('\n'.join(Q))
        
    with open(r'{}.json'.format(Path(ROOT_DIR,model_name)), 'w') as fp:
        json.dump(slipping_probabilities, fp)
        
    with open(r'{}_mle.json'.format(Path(ROOT_DIR,model_name)), 'w') as fp:
        json.dump(slipping_estimates, fp)
        
    print('Exported model with name "{}"'.format(model_name))
    
        
    
def gen_pMDP_random(terrain, model_name,
                   loc_package, loc_warehouse, reward, nodrn=True):
    
    print('')
    
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
    
    # Export to PRISM file
    with open(r'{}'.format(prism_path), 'w') as fp:
        fp.write('\n'.join(Q))
        
    print('Exported model with name "{}"'.format(model_name))
    
    if nodrn is False:
    
        drn_path   = str(Path(ROOT_DIR, str(model_name)+'_conv.drn'))    
    
        # Convert from PRISM to DRN file
        formula = 'Rmin=? [F \"goal\"]'
        
        program = stormpy.parse_prism_program(prism_path)
        properties = stormpy.parse_properties(formula, program)
        model = stormpy.build_parametric_model(program, properties)
        
        stormpy.export_to_drn(model, drn_path)
    
        path = drn_path
    else:
        path = prism_path
    
    return path


def xyz_to_plain_state(x,y,z,X,Y):
    
    s = int(x + y*X + z*(X*Y))
        
    return s

def gen_pMDP_random_drn(N, terrain, model_name,
                        loc_package, loc_warehouse, reward, slipmode='fix'):
    
    '''
    Generates a parametric DTMC for the grid world example of arbitrary size,
    under a random policy.
    Exports directly to a DRN file to avoid costly conversion by PRISM parser!
    '''    
    
    print('')
    
    # Determine size of the grid
    (Y,X) = terrain.shape  
    (x_init, y_init) = (0,0)

    # Note: first element is y, second element is x
    dct = {'east':  (0,  1),
           'south':  (1,  0),
           'west':  (0, -1),
           'north':  (-1, 0)
            }
    
    Q = ['// Exported by generate_slipgrid_pmc.py',
         '// Original model type: DTMC',
         '@type: DTMC',
         '@parameters']
    
    Q += [" ".join(['v{}'.format(t) for t in np.unique(terrain)])]
        
    Q += ['@reward_models\n',
          '@nr_states',
          str(int(X*Y*2-1)),
          '@nr_choices',
          str(int(X*Y*2-1)),
          '@model']
    
    s_package = xyz_to_plain_state(loc_package[0], loc_package[1], 0, X, Y)
    
    for z in [0,1]:
        for y,row in enumerate(terrain):
            for x,ter in enumerate(row):
            
                if x == x_init and y == y_init and z == 0:
                    label = 'init'
                    
                elif (x,y) == loc_warehouse and z == 1:
                    label = 'goal'
                    
                else:
                    label = ''
                    
                s = xyz_to_plain_state(x, y, z, X, Y)
                
                if s == s_package:
                    continue
                elif s > s_package:
                    s -= 1
                
                Q += ['state {} [0] {}'.format(s, label)]
                    
                Q += ['\taction 0 [{}]'.format(0 if label == 'goal' else reward)]
            
                # Set goal state as terminal state
                if label == 'goal':
                    Q += ['\t\t{} : 1'.format(s)]
                    
                else:
                    
                    #####
                    
                    
                    # Slip = remain on same locatoin
                    if slipmode == 'fix':
                        
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
                        
                        # Slipping
                        Q += ['\t\t{} : v{}'.format(s, ter)]
                    
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
                                z_new = 1
                            else:
                                z_new = z
                                
                            s_new = xyz_to_plain_state(x_new, y_new, z_new, X, Y)        
                            
                            if s_new > s_package:
                                s_new -= 1
                            
                            Q += ['\t\t{} : 1/{}*(1-v{})'.format(s_new, int(moves), ter)]
                           
                            
                    # Slip = move two places instead of one
                    elif slipmode == 'double':
                        
                        # In this case, only move right/down and wrap around
                        # if the agent goes out of bounds.
                        
                        moves = 2
                        Q_sub = {}
                        
                        # Iterate over possible moves
                        for lab, move in dct.items():
                            
                            # Only move right/down
                            if lab in ['west', 'north']:
                                continue
                            
                            # Only slip when moving down
                            if lab == 'south':
                                slip = True
                            else:
                                slip = False
                            
                            ### Normal move
                            # Compute new position
                            x_new = x + move[1]
                            y_new = y + move[0]
                            
                            # If out of bounds, wrap around
                            if x_new < 0:
                                x_new += X
                            elif x_new >= X:
                                x_new -= X
                                
                            if y_new < 0:
                                y_new += Y
                            elif y_new >= Y:
                                y_new -= Y
                            
                            # Check if we have reached the package
                            if (x_new, y_new) == loc_package:
                                z_new = 1
                            else:
                                z_new = z
                                
                            s_new = xyz_to_plain_state(x_new, y_new, z_new, X, Y)        
                            
                            if s_new > s_package:
                                s_new -= 1
                            
                            if slip:
                                Q_sub[s_new] = '\t\t{} : 1/{}*(1-v{})'.format(s_new, int(moves), ter)
                            else:
                                Q_sub[s_new] = '\t\t{} : 1/{}'.format(s_new, int(moves))
                        
                            # Slipping move
                            # Compute new position
                            x_slip = x + int(move[1] * 2)
                            y_slip = y + int(move[0] * 2)
                            
                            # Check if out of bounds
                            if x_slip < 0:
                                x_slip += X
                            elif x_slip >= X:
                                x_slip -= X
                                
                            if y_slip < 0:
                                y_slip += Y
                            elif y_slip >= Y:
                                y_slip -= Y
                            
                            # Check if we have reached the package
                            if (x_slip, y_slip) == loc_package:
                                z_slip = 1
                            else:
                                z_slip = z
                                
                            s_slip = xyz_to_plain_state(x_slip, y_slip, z_slip, X, Y)        
                            
                            if s_slip > s_package:
                                s_slip -= 1
                            
                            if slip:
                                Q_sub[s_slip] = '\t\t{} : 1/{}*(v{})'.format(s_slip, int(moves), ter)
                            else:
                                Q_sub[s_new] = '\t\t{} : 1/{}'.format(s_new, int(moves))
                        
                        for key in np.sort(list(Q_sub.keys())):
                            Q += [Q_sub[key]]
                           
                    else:
                        print('Unknown slip mode. Exit.')
                        assert False
                        
                    
                    
    drn_path   = str(Path(ROOT_DIR, str(model_name)+'.drn'))
    
    # Export to PRISM file
    with open(r'{}'.format(drn_path), 'w') as fp:
        fp.write('\n'.join(Q))
        
    print('Exported model directly to DRN with name "{}"'.format(model_name))
    
    return drn_path
    

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
    'v1': 0.25,
    'v2': 0.40,
    'v3': 0.45,
    'v4': 0.50,
    'v5': 0.35
    }

# Generate rough estimate of the slipping probabilities
N = {
     'v1': 12,
     'v2': 18*2,
     'v3': 15*2,
     'v4': 30*2,
     'v5': 11*2
     }

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
    [0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [0, -1, 2, 2, 2]
    ])

model_name = "models/slipgrid/dummy"

loc_package   = (4, 1)
loc_warehouse = (1, 4)

gen_pMC(N, slipping_probabilities, policy_before, policy_after,
        terrain, model_name, loc_package, loc_warehouse)

# %%

##########################################
# Generate other, random slipgrids

cases = [
    (10,    10),
    (50,    50),
    (50,    100),
    (50,    1000),
    (100,   100),
    (100,   1000),
    (200,   100),
    (200,   1000),
    (200,   10000),
    (400,   100),
    (400,   1000),
    (400,   10000),
    (800,   100)
    ]

p_range = [0.10, 0.20]

# Number of parameters to estimate probabilities with
Nmin = 500
Nmax = 1000

ITERS = 1

BASH_FILE = ["#!/bin/bash",
             "cd ..;",
             'echo -e "START GRID WORLD EXPERIMENTS...";']

num_derivs = 1

for (Z,V) in cases:
    for mode in ['fix', 'double']:
      
        N = np.array(np.random.uniform(low=Nmin, high=Nmax, size=V), dtype=int)
          
        for seed in range(ITERS):
                    
            np.random.seed(0)
                    
            model_name = "models/slipgrid/{}_pmc_size={}_params={}_seed={}".format(mode,Z,V,seed)
            
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
            for i,(v,p) in enumerate(slipping_probabilities.items()):
                
                slipping_estimates[v] = [max(min(np.random.binomial(N[i], p) / N[i], 1-delta), delta), int(N[i])]
               
            json_path  = str(Path(ROOT_DIR, str(model_name)+'.json'))
            with open(r'{}.json'.format(str(Path(ROOT_DIR, str(model_name)))), 'w') as fp:
                json.dump(slipping_probabilities, fp)
                
            json_mle_path  = str(Path(ROOT_DIR, str(model_name)+'_mle.json'))
            with open(r'{}_mle.json'.format(str(Path(ROOT_DIR, str(model_name)))), 'w') as fp:
                json.dump(slipping_estimates, fp)
            
            # Scale reward with the size of the grid world
            reward = 10**(-math.floor(math.log(Z**2, 10)))
            
            path = gen_pMDP_random(
                terrain, model_name, loc_package, loc_warehouse, reward, nodrn=True)
            
            drn_path = gen_pMDP_random_drn(N, terrain, model_name,
                                    loc_package, loc_warehouse, reward, slipmode = mode)
            
            command = ["timeout 3600s python3 run_cav23.py",
                       '--instance "grid({},{})"'.format(Z,V),
                       "--model '{}'".format(drn_path),
                       "--parameters '{}'".format(json_mle_path),
                       "--formula 'Rmin=? [F \"goal\"]'",
                       "--pMC_engine 'spsolve'",
                       "--validate_delta -0.001",
                       "--output_folder 'output/slipgrid/'",
                       "--num_deriv {}".format(num_derivs),
                       "--explicit_baseline",
                       "--robust_bound 'lower'",
                       "--scale_reward"]
            
            if (Z >= 100 and mode == 'fix') or (Z >= 200 and mode == 'double'):
                command += ["--no_prMC"]
            
            BASH_FILE += [" ".join(command)+";"]
        
BASH_FILE += ["#", "python3 create_table.py --folder 'output/slipgrid/' --table_name 'tables/slipgrid_table'"]
        
# Export bash file to perform grid world experiments
with open(str(Path(ROOT_DIR,'experiments/grid_world.sh')), 'w') as f:
    f.write("\n".join(BASH_FILE))
    
    
    
    
    











# model_name = "models/slipgrid/input/pmdp_size={}_params={}".format(Z,V)
        # gen_pMDP(N, terrain, model_name,
        #          loc_package, loc_warehouse)
   
# def gen_pMDP(N, terrain, model_name,
#              loc_package, loc_warehouse):
    
#     '''
#     Generates a parametric MDP for the grid world example of arbitrary size.
#     Exports to a PRISM model file.
#     '''
    
#     # Determine size of the grid
#     (Y,X) = terrain.shape  

#     # Note: first element is y, second element is x
#     dct = {'east':  (0,  1),
#            'south':  (1,  0),
#            'west':  (0, -1),
#            'north':  (-1, 0)
#             }
    
#     Q = ['mdp\n']
    
#     for t in np.unique(terrain):
#         Q += ['const double v{};'.format(t)]
        
#     Q += ['\n module grid',
#           '    x : [0..4] init 0;',
#           '    y : [0..4] init 0;',
#           '    z : [0..1] init 0;']
    
#     for y,row in enumerate(terrain):
#         for x,ter in enumerate(row):
            
#             # Iterate over possible moves
#             for lab, move in dct.items():
                
#                 # Compute new position
#                 x_new = x + move[1]
#                 y_new = y + move[0]
                
#                 # Check if out of bounds
#                 if x_new < 0 or x_new >= X:
#                     continue
#                 if y_new < 0 or y_new >= Y:
#                     continue
                
#                 # Check if we have reached the package
#                 if (x_new, y_new) == loc_package:
#                     z_new = 'z+1'
#                 else:
#                     z_new = 'z'
                    
#                 # Transition for before picking up the package
#                 Q += ["    [{}] (x={}) & (y={}) & (z=0) -> 1-v{}: (x'={}) & (y'={}) & (z'={}) + v{}: (x'=x) & (y'=y) & (z'=z);".format(lab, x, y,
#                                                                                                                                    ter, x_new, y_new, z_new,  
#                                                                                                                                    ter)]
       
#                 # Now add transitions after picking up the package
#                 Q += ["    [{}] (x={}) & (y={}) & (z=1) -> 1-v{}: (x'={}) & (y'={}) & (z'=z) + v{}: (x'=x) & (y'=y) & (z'=z);".format(lab, x, y,
#                                                                                                                                    ter, x_new, y_new,  
#                                                                                                                                    ter)]
            
#     Q += ['endmodule\n',
#           '// reward structure (number of steps to reach the target)',
#           'rewards',
#           '    [east] true : 1;',
#           '    [south] true : 1;',
#           '    [west] true : 1;',
#           '    [north] true : 1;',
#           'endrewards\n',
#           'label "goal" = x={} & y={} & z=1;'.format(loc_warehouse[0], loc_warehouse[1])
#           ]
    
#     # Print to file
#     with open(r'{}.nm'.format(str(Path(ROOT_DIR, str(model_name)))), 'w') as fp:
#         fp.write('\n'.join(Q))
        
#     print('Exported model with name "{}"'.format(str(Path(ROOT_DIR, str(model_name)))))