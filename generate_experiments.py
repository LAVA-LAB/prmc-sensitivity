import numpy as np
import json
import math
import os
from pathlib import Path

from core.experiments.generate_slipgrids import generate_slipgrid, \
    generate_pmc_random_drn, generate_pmc_learning_drn

# Load PRISM model with STORM
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# %%
#####################################################
# Generate shell execution file for benchmark suite #

brp = {
    0: {'model':    "models/pdtmc/brp16_2.pm",      
        'formula':  "P=? [ F s=5 ]",       
        'extra':    "--default_valuation 0.9 --goal_label \"{'(s = 5)'}\" --validate_delta 1e-3"},
    1: {'model':    "models/pdtmc/brp32_3.pm",      
        'formula':  "P=? [ F s=5 ]",       
        'extra':    "--default_valuation 0.9 --goal_label \"{'(s = 5)'}\" --validate_delta 1e-3"},
    2: {'model':    "models/pdtmc/brp64_4.pm",      
        'formula':  "P=? [ F s=5 ]",       
        'extra':    "--default_valuation 0.9 --goal_label \"{'(s = 5)'}\" --validate_delta 1e-3"},
    3: {'model':    "models/pdtmc/brp512_5.pm",      
        'formula':  "P=? [ F s=5 ]",       
        'extra':    "--default_valuation 0.9 --goal_label \"{'(s = 5)'}\" --validate_delta 1e-3"},
    4: {'model':    "models/pdtmc/brp1024_6.pm",      
        'formula':  "P=? [ F s=5 ]",       
        'extra':    "--default_valuation 0.9 --goal_label \"{'(s = 5)'}\" --validate_delta 1e-3"},
    }

crowds = {
    0: {'model':    "models/pdtmc/crowds3_5.pm",      
        'formula':  "P=? [F \"observe0Greater1\" ]",       
        'extra':    "--goal_label \"{'observe0Greater1'}\" --validate_delta 1e-3"},
    1: {'model':    "models/pdtmc/crowds6_5.pm",      
        'formula':  "P=? [F \"observe0Greater1\" ]",       
        'extra':    "--goal_label \"{'observe0Greater1'}\" --validate_delta 1e-3"},
    2: {'model':    "models/pdtmc/crowds10_5.pm",      
        'formula':  "P=? [F \"observe0Greater1\" ]",       
        'extra':    "--goal_label \"{'observe0Greater1'}\" --validate_delta 1e-3"},
    }

nand = {
    0: {'model':    "models/pdtmc/nand2_4.pm",      
        'formula':  "P=? [F \"target\" ]",
        'extra':    "--parameters 'models/pdtmc/nand.json' --goal_label \"{'target'}\" --validate_delta 1e-2"},
    1: {'model':    "models/pdtmc/nand5_10.pm",      
        'formula':  "P=? [F \"target\" ]",
        'extra':    "--parameters 'models/pdtmc/nand.json' --goal_label \"{'target'}\" --validate_delta 1e-2"},
    2: {'model':    "models/pdtmc/nand10_15.pm",      
        'formula':  "P=? [F \"target\" ]",
        'extra':    "--parameters 'models/pdtmc/nand.json' --goal_label \"{'target'}\" --validate_delta 1e-2"},
    }

virus = {
    0: {'model':    'models/pmdp/virus/virus.pm',      
        'formula':  'R{"attacks"}max=? [F s11=2 ]',
        'extra':    "--default_valuation 0.1 --validate_delta 1e-3"}
    }

wlan = {
    0: {'model':    'models/pmdp/wlan/wlan0_param.nm',      
        'formula':  'R{"time"}max=? [ F s1=12 | s2=12 ]',
        'extra':    "--default_valuation 0.001 --validate_delta 1e-3"}  
    }

csma = {
    0: {'model':    'models/pmdp/CSMA/csma2_4_param.nm',      
        'formula':  'R{"time"}max=? [ F "all_delivered" ]',
        'extra':    "--default_valuation 0.05 --validate_delta 1e-3"}  
    }

coin = {
    0: {'model':    'models/pmdp/coin/coin4.pm',      
        'formula':  'Pmin=? [ F "all_coins_equal_1" ]',
        'extra':    "--default_valuation 0.4 --goal_label \"{'all_coins_equal_1'}\" --validate_delta 1e-3"}  
    }

# Gives unbounded model
drone_sttt = {
    0: {'model':    'models/sttt-drone/drone_model.nm',      
        'formula':  'Pmax=? [F attarget ]',
        'extra':    "--default_valuation 0.07692307692 --goal_label \"{'(((x > (15 - 2)) & (y > (15 - 2))) & (z > (15 - 2)))'}\" --validate_delta 1e-3"}  
    }

maze = {
    0: {'model':    'models/pomdp/maze/maze_simple_extended_m5.drn',      
        'formula':  'Rmin=? [F "goal"]',
        'extra':    "--validate_delta 1e-3"}  
    }

# Segmentation fault
network = {
    0: {'model':    'models/pomdp/network/network2K-20_T-8_extended-simple_full.drn',      
        'formula':  'R{"dropped_packets"}=? [F "goal"]',
        'extra':    "--goal_label \"{'goal'}\" --validate_delta 1e-3export"}  
    }

f = 'P=? ["notbad" U "goal"]'
e = "--goal_label \"{'goal','notbad'}\" --validate_delta 1e-3"

drone = {
    0: {'model':    'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn',      
        'formula':  f,
        'extra':    e},
    1: {'model':    'models/pomdp/drone/pomdp_drone_4-2-mem5-simple.drn',      
        'formula':  f,
        'extra':    e}  
    }

f = 'P=? [F "goal"]'
e = "--default_valuation 0.01 --goal_label \"{'goal'}\" --validate_delta 1e-3"  

satellite = {
    0: {'model':    'models/satellite/pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06_sparse_full.drn',      
        'formula':  f,
        'extra':    e},
    1: {'model':    'models/satellite/pomdp_prob_36_sat_065_dist_1_obs_diff_orb_len_40.drn',      
        'formula':  f,
        'extra':    e}  
    }

suites = {
    'brp': brp, 
    'crowds': crowds, 
    'nand': nand, 
    'virus': virus, 
    'wlan': wlan, 
    'csma': csma, 
    'coin': coin,
    'maze': maze, 
    'drone': drone,
    'satellite': satellite,
    }

# Main configurations
OUTPUT_FOLDER           = "output/benchmarks"
OUTPUT_FOLDER_PARTIAL   = "output/benchmarks_partial" 

TABLE_NAME              = "output/benchmarks_table"
TABLE_NAME_PARTIAL      = "output/benchmarks_partial_table"

BASH_OUT_FILE           = "experiments/benchmarks.sh"
BASH_OUT_FILE_PARTIAL   = "experiments/benchmarks_partial.sh"

NUMBER_DERIVATIVES      = [1, 10]

# Write experiment bash file
prefix_pmc  = "timeout 3600s python3 run_pmc.py"
prefix_prmc = "timeout 3600s python3 run_prmc.py"

OUTPUT                  = "--output_folder '"+str(OUTPUT_FOLDER)+"'"
OUTPUT_PARTIAL          = "--output_folder '"+str(OUTPUT_FOLDER_PARTIAL)+"'"

text = \
['''#!/bin/bash
cd ..;
echo -e "START BENCHMARK SUITE...";''']

text_partial = \
['''#!/bin/bash
cd ..;
echo -e "START BENCHMARK SUITE...";''']

for name,suite in suites.items():
    text += ['\n# BENCHMARKS FOR {}'.format(str(name))]
    text_partial += ['\n# BENCHMARKS FOR {}'.format(str(name))]
    
    for k in NUMBER_DERIVATIVES:  
      for mode in ['full', 'partial']:
        for i,exp in suite.items():
        
            if mode == 'partial' and i > 0:
                break
        
            if mode == 'full':
                output = OUTPUT
            else:
                output = OUTPUT_PARTIAL
        
            # Write command for full benchmark suite
            model_string = "--model '{}'".format(str(exp['model']))
            formula_string = "--formula '{}'".format(str(exp['formula']))
            additional_string = exp['extra']
            
            suffix  = "--num_deriv {} --explicit_baseline --no_gradient_validation".format(int(k))
            
            # Add command for pMC benchmark
            command = " ".join([prefix_pmc, model_string, formula_string, additional_string, output, suffix])+';'
            
            if mode == 'full':
                text += [command]
            else:
                text_partial += [command]
            
            
            # Add command for prMC benchmark
            command = " ".join([prefix_prmc, model_string, formula_string, additional_string, output, suffix])+';'
            
            if mode == 'full':
                text += [command]
            else:
                text_partial += [command]
            
            suffix  = "--num_deriv {} --explicit_baseline --no_gradient_validation --no_par_dependencies".format(int(k))
            
            # Add command for prMC benchmark, without parameter dependencies
            command = " ".join([prefix_prmc, model_string, formula_string, additional_string, output, suffix])+';'
            
            if mode == 'full':
                text += [command]
            else:
                text_partial += [command]
            

text += \
['''
 python3 create_table.py --folder '{}' --table_name '{}' --mode benchmark'''.format(OUTPUT_FOLDER, TABLE_NAME)]
 
# Save to file
with open(r'{}'.format(Path(ROOT_DIR, BASH_OUT_FILE)), 'w') as fp:
    fp.write('\n'.join(text))
 
print('\nExported full benchmark suite')   
 
text_partial += \
['''
 python3 create_table.py --folder '{}' --table_name '{}' --mode benchmark'''.format(OUTPUT_FOLDER_PARTIAL, TABLE_NAME_PARTIAL)]

# Save to file
with open(r'{}'.format(Path(ROOT_DIR, BASH_OUT_FILE_PARTIAL)), 'w') as fp:
    fp.write('\n'.join(text_partial))

print('\nExported partial benchmark suite')

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

generate_slipgrid(ROOT_DIR, N, slipping_probabilities, policy_before, policy_after,
        terrain, model_name, loc_package, loc_warehouse)

# Generate shell file to run this experiment

text = '''#!/bin/bash
cd ..;
echo -e "CREATE DATA FOR MOTIVATING EXAMPLE IN PAPER...";
#
python3 run_pmc.py --model 'models/slipgrid/dummy.nm' --parameters 'models/slipgrid/dummy.json' --formula 'Rmin=? [F "goal"]' --validate_delta 0.001 --output_folder 'output/motivating_example/' --num_deriv 4;
#
python3 run_pmc.py --model 'models/slipgrid/dummy.nm' --parameters 'models/slipgrid/dummy_mle.json' --formula 'Rmin=? [F "goal"]' --validate_delta 0.001 --output_folder 'output/motivating_example/' --num_deriv 4;
#
python3 run_prmc.py --model 'models/slipgrid/dummy.nm' --parameters 'models/slipgrid/dummy_mle.json' --formula 'Rmin=? [F "goal"]' --validate_delta 0.001 --output_folder 'output/motivating_example/' --num_deriv 4;
'''

# Save to file
with open(r'{}'.format(Path(ROOT_DIR, "experiments/motivating_example_paper.sh")), 'w') as fp:
    fp.write(text)

# %%

###################################################
# Generate randomized slipgrid of different sizes #

cases = [
    (10,    10),
    (20,    100),
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
    (800,   100),
    (800,   1000),
    (800,   10000)
    ]

cases_partial = [
    (10,    10),
    (20,    100),
    (50,    50),
    (50,    100),
    (50,    1000)
    ]

slipping_prob_range = [0.10, 0.20]

# Number of parameters to estimate probabilities with
Nmin = 500
Nmax = 1000

number_derivatives = [1, 10]

BASH_FILE = {
    'full': ["#!/bin/bash",
             "cd ..;",
             'echo -e "START GRID WORLD EXPERIMENTS (FULL VERSION)...";'],
    'partial': ["#!/bin/bash",
             "cd ..;",
             'echo -e "START GRID WORLD EXPERIMENTS (FULL VERSION)...";'],
    }

BASH_PARTIAL = []

prefix_pmc  = ["timeout 3600s python3 run_pmc.py"]
prefix_prmc = ["timeout 3600s python3 run_prmc.py"]

# Create slipgrid model files
for (Z,V) in cases:

    np.random.seed(0)
            
    model_name = "models/slipgrid/pmc_size={}_params={}".format(Z,V)
    
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
        'v'+str(i): np.random.uniform(slipping_prob_range[0], 
                                      slipping_prob_range[1]) for i in range(V)
        }
        
    # Minimum transition probability estimate
    delta = 1e-4
    
    # Obtain point estimates for each of the transition probabilities
    slipping_estimates = {}
    N = np.array(np.random.uniform(low=Nmin, high=Nmax, size=V), dtype=int)
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
    
    drn_path = generate_pmc_random_drn(ROOT_DIR, N, terrain, model_name,
                            loc_package, loc_warehouse, reward, slipmode = 'double')

    command = ['--instance "grid({},{})"'.format(Z,V),
               "--model '{}'".format(drn_path),
               "--parameters '{}'".format(json_mle_path),
               "--formula 'Rmin=? [F \"goal\"]'",
               "--robust_bound 'upper'"]
    
    for mode in ['full', 'partial']:
      # If mode is partial, check if this case should actually be included
      if mode == 'partial' and (Z,V) not in cases_partial:
        continue
        
      # For all number of derivatives
      for num_derivs in number_derivatives:
          
        numderiv = ["--num_deriv {}".format(num_derivs)]
        
        # Set output folder
        if mode == 'full':
            outfolder = ["--output_folder 'output/slipgrid/'"]
        else:
            outfolder = ["--output_folder 'output/slipgrid_partial/'"]
        
        suffix = []

        # Depending on the number of derivatives, set certain arguments
        if num_derivs > 1:
            # If multiple derivatives are computed, skip the validation
            suffix += ["--no_gradient_validation"]
        else:
            # Only perform the explicit baseline (computing all derivatives)
            # if 1 derivative is computed
            suffix += ["--explicit_baseline"]
            
        BASH_FILE[mode] += [" ".join(prefix_pmc + command + numderiv + outfolder + suffix)+";"]
        
        # Only perform experiments for prMCs with grids up to size 200x200
        if Z <= 200:
            BASH_FILE[mode] += [" ".join(prefix_prmc + command + numderiv + outfolder + suffix)+";"]

BASH_FILE['full']    += ["#", "python3 create_table.py --folder 'output/slipgrid/' --table_name 'output/slipgrid_table' --mode 'gridworld'"]  

# Export bash file to perform grid world experiments
with open(str(Path(ROOT_DIR,'experiments/grid_world.sh')), 'w') as f:
    f.write("\n".join(BASH_FILE['full']))

print('\nExported full grid world experiment shell script')

BASH_FILE['partial'] += ["#", "python3 create_table.py --folder 'output/slipgrid_partial/' --table_name 'output/slipgrid_partial_table' --mode 'gridworld'"]        
        
# Export bash file to perform grid world experiments
with open(str(Path(ROOT_DIR,'experiments/grid_world_partial.sh')), 'w') as f:
    f.write("\n".join(BASH_FILE['partial']))

print('\nExported partial grid world experiment shell script')

# %%

###################################################
# Generate slipgrids for learning framework       #

np.random.seed(0)

cases = [
    (20,   100),
    ]

slipping_prob_range = [0.10, 0.50]

# Number of samples to estimate probabilities with
Nmin = 100
Nmax = 100

iterations = [0]
BIAS = True

for (Z,V) in cases:      
    N = np.array(np.random.uniform(low=Nmin, high=Nmax, size=V), dtype=int)
      
    np.random.seed(0)
                
    model_name = "models/slipgrid_learning/pmc_size={}_params={}".format(Z,V)
    
    # By default, put package in top right corner and warehouse in bottom left
    loc_package   = (Z-1, 0)
    loc_warehouse = (0, Z-1)
    
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
    
    # print('Terrain:')
    # print(terrain)
    # for w in range(V):
    #     print(w,':',len(np.where(terrain == w)[0]))
                
    # Set slipping probabilities (v1 corresponds with terrin type 1)
    slipping_probabilities = {
        'v'+str(i): np.random.uniform(slipping_prob_range[0], 
                                      slipping_prob_range[1]) for i in range(V)
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
    
    drn_path = generate_pmc_learning_drn(ROOT_DIR, N, terrain, model_name,
                            loc_package, loc_warehouse, reward, slipmode = 'double')

# %%

########################################################################
# Generate combined shell scripts to run partial/full experiment sets  #

# FULL BENCHMARK SET
SHELL = ["#!/bin/bash",
         'echo -e "START FULL SET OF EXPERIMENTS...";',
         '',
         'echo -e "\\nStart grid world experiments...\\n\\n";',
         'bash grid_world.sh;',
         '',
         'echo -e "\\nStart other benchmarks...\\n\\n";',
         'bash benchmarks.sh;',
         '',
         'echo -e "\\nStart learning experiments...\\n\\n";',
         'cd ..;',
         '''python3 run_learning.py --instance gridworld --model models/slipgrid_learning/pmc_size=20_params=100.drn --parameters models/slipgrid_learning/pmc_size=20_params=100_mle.json --formula 'Rmin=? [F "goal"]' --output_folder 'output/learning/' --num_deriv 1 --robust_bound 'upper' --uncertainty_model 'Hoeffding' --true_param_file models/slipgrid_learning/pmc_size=20_params=100.json --learning_iterations 10 --learning_steps 1000 --learning_samples_per_step 25;''',
         '''python3 run_learning.py --instance drone --model models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn --formula 'P=? ["notbad" U "goal"]' --output_folder 'output/learning/' --num_deriv 1 --robust_bound 'upper' --uncertainty_model 'Hoeffding' --goal_label "{'goal','notbad'}" --default_sample_size 100 --learning_iterations 10 --learning_steps 1000 --learning_samples_per_step 25;'''
         ]

# Export bash file to perform grid world experiments
with open(str(Path(ROOT_DIR,'experiments/all_experiments_full.sh')), 'w') as f:
    f.write("\n".join(SHELL))

print('\nExported shell scripts to run all experiments (full version)')

# PARTIAL BENCHMARK SET
SHELL = ["#!/bin/bash",
         'echo -e "START FULL SET OF EXPERIMENTS...";',
         '',
         'echo -e "\\nStart grid world experiments...\\n\\n";',
         'bash grid_world_partial.sh;',
         '',
         'echo -e "\\nStart other benchmarks...\\n\\n";',
         'bash benchmarks_partial.sh;',
         '',
         'echo -e "\\nStart learning experiments...\\n\\n";',
         'cd ..;',
         '''python3 run_learning.py --instance gridworld --model models/slipgrid_learning/pmc_size=20_params=100.drn --parameters models/slipgrid_learning/pmc_size=20_params=100_mle.json --formula 'Rmin=? [F "goal"]' --output_folder 'output/learning/' --num_deriv 1 --robust_bound 'upper' --uncertainty_model 'Hoeffding' --true_param_file models/slipgrid_learning/pmc_size=20_params=100.json --learning_iterations 1 --learning_steps 100 --learning_samples_per_step 25;''',
         '''python3 run_learning.py --instance drone --model models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn --formula 'P=? ["notbad" U "goal"]' --output_folder 'output/learning/' --num_deriv 1 --robust_bound 'upper' --uncertainty_model 'Hoeffding' --goal_label "{'goal','notbad'}" --default_sample_size 100 --learning_iterations 1 --learning_steps 100 --learning_samples_per_step 25;'''
         ]

# Export bash file to perform grid world experiments
with open(str(Path(ROOT_DIR,'experiments/all_experiments_partial.sh')), 'w') as f:
    f.write("\n".join(SHELL))

print('\nExported shell scripts to run all experiments (partial version)')
