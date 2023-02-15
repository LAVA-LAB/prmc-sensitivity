# %run "~/documents/CAV23/prmdp-sensitivity/generate_slipgrid_pmc_learning.py"

import numpy as np
import json
import stormpy
import os
import math
from pathlib import Path

from datetime import datetime

# Load PRISM model with STORM
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    
##########################################
# Generate other, random slipgrids

np.random.seed(0)

cases = [
    (20,    100),
    ]

p_range = [0.10, 0.50]

# Number of parameters to estimate probabilities with
Nmin = 100
Nmax = 100

ITERS = 1
BIAS = True

NUM = [10]
dt = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")

for num_derivs in NUM:
  for (Z,V) in cases:
    for mode in ['double']:
      
        N = np.array(np.random.uniform(low=Nmin, high=Nmax, size=V), dtype=int)
          
        for seed in range(ITERS):
                    
            np.random.seed(0)
                    
            model_name = "models/slipgrid_learning/{}_pmc_size={}_params={}_seed={}".format(mode,Z,V,seed)
            
            # By default, put package in top right corner and warehouse in bottom left
            loc_package   = (Z-2, 1)
            loc_warehouse = (1, Z-2)
            
            if V > Z**2:
                print('Skip configuration, because no. params exceed no. states.')
                continue
            
            # Set ID's of terrain types
            if BIAS:
                
                order = np.arange(0, V)
                np.random.shuffle(order)
                
                listA = np.random.randint(low=0, high=10, size=int(Z**2/2))
                terrainA = order[listA]
                listB = np.random.randint(low=10, high=30, size=int(Z**2/4))
                terrainB = order[listB]
                listC = np.random.randint(low=30, high=V, size=int(Z**2/4))
                terrainC = order[listC]
            
                terrain = np.concatenate((terrainA, terrainB, terrainC))
                np.random.shuffle(terrain)
                terrain = np.reshape(terrain, (Z,Z))
            
            else:
                terrain = np.random.randint(low = 0, high = V, size=(Z,Z))
            
            print('Terrain:')
            print(terrain)
            for w in range(V):
                print(w,':',len(np.where(terrain == w)[0]))
                        
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
            
            drn_path = gen_pMDP_random_drn(N, terrain, model_name,
                                    loc_package, loc_warehouse, reward, slipmode = mode)
    