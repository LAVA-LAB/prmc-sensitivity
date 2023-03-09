import os

def run_experiment(exp, prefix, output, suffix):
    
    model_string = "--model '{}'".format(str(exp['model']))
    formula_string = "--formula '{}'".format(str(exp['formula']))
    additional_string = exp['extra']
    
    string = " ".join([prefix, model_string, formula_string, additional_string, output, suffix])
    os.system(string)

    # import subprocess
    # subprocess.Popen(string, shell=True).wait()    


prefix_pmc  = "timeout 3600s python3 run_pmc.py"
prefix_prmc = "timeout 3600s python3 run_prmc.py"
output  = "--output_folder 'output/test_results/'"
suffix  = "--num_deriv 1 --validate_delta 1e-5" #" --explicit_baseline"

brp = {
    0: {'model':    "models/pdtmc/brp16_2.pm",      
        'formula':  "P=? [ F s=5 ]",       
        'extra':    "--default_valuation 0.9 --goal_label \"{'(s = 5)'}\""},
    1: {'model':    "models/pdtmc/brp32_3.pm",      
        'formula':  "P=? [ F s=5 ]",       
        'extra':    "--default_valuation 0.9 --goal_label \"{'(s = 5)'}\""},
    2: {'model':    "models/pdtmc/brp64_4.pm",      
        'formula':  "P=? [ F s=5 ]",       
        'extra':    "--default_valuation 0.9 --goal_label \"{'(s = 5)'}\""},
    3: {'model':    "models/pdtmc/brp512_5.pm",      
        'formula':  "P=? [ F s=5 ]",       
        'extra':    "--default_valuation 0.9 --goal_label \"{'(s = 5)'}\""},
    # 4: {'model':    "models/pdtmc/brp1024_6.pm",      
    #     'formula':  "P=? [ F s=5 ]",       
    #     'extra':    "--default_valuation 0.9 --goal_label \"{'(s = 5)'}\""},
    }

crowds = {
    0: {'model':    "models/pdtmc/crowds3_5.pm",      
        'formula':  "P=? [F \"observe0Greater1\" ]",       
        'extra':    "--goal_label \"{'observe0Greater1'}\""},
    1: {'model':    "models/pdtmc/crowds6_5.pm",      
        'formula':  "P=? [F \"observe0Greater1\" ]",       
        'extra':    "--goal_label \"{'observe0Greater1'}\""},
    # 2: {'model':    "models/pdtmc/crowds10_5.pm",      
    #     'formula':  "P=? [F \"observe0Greater1\" ]",       
    #     'extra':    "--goal_label \"{'observe0Greater1'}\""},
    }

nand = {
    0: {'model':    "models/pdtmc/nand2_4.pm",      
        'formula':  "P=? [F \"target\" ]",
        'extra':    "--parameters 'models/pdtmc/nand.json' --goal_label \"{'target'}\""},
    1: {'model':    "models/pdtmc/nand5_10.pm",      
        'formula':  "P=? [F \"target\" ]",
        'extra':    "--parameters 'models/pdtmc/nand.json' --goal_label \"{'target'}\""},
    # 2: {'model':    "models/pdtmc/nand10_15.pm",      
    #     'formula':  "P=? [F \"target\" ]",
    #     'extra':    "--parameters 'models/pdtmc/nand.json' --goal_label \"{'target'}\""},
    }

virus = {
    0: {'model':    'models/pmdp/virus/virus.pm',      
        'formula':  'R{"attacks"}max=? [F s11=2 ]',
        'extra':    "--default_valuation 0.1"}
    }

wlan = {
    0: {'model':    'models/pmdp/wlan/wlan0_param.nm',      
        'formula':  'R{"time"}max=? [ F s1=12 | s2=12 ]',
        'extra':    "--default_valuation 0.01"}  
    }

csma = {
    0: {'model':    'models/pmdp/CSMA/csma2_4_param.nm',      
        'formula':  'R{"time"}max=? [ F "all_delivered" ]',
        'extra':    "--default_valuation 0.1"}  
    }

coin = {
    0: {'model':    'models/pmdp/coin/coin4.pm',      
        'formula':  'Pmin=? [ F "all_coins_equal_1" ]',
        'extra':    "--default_valuation 0.4 --goal_label \"{'all_coins_equal_1'}\""}  
    }

# Gives unbounded model
drone_sttt = {
    0: {'model':    'models/sttt-drone/drone_model.nm',      
        'formula':  'Pmax=? [F attarget ]',
        'extra':    "--default_valuation 0.07692307692 --goal_label \"{'(((x > (15 - 2)) & (y > (15 - 2))) & (z > (15 - 2)))'}\""}  
    }

maze = {
    0: {'model':    'models/pomdp/maze/maze_simple_extended_m5.drn',      
        'formula':  'Rmin=? [F "goal"]',
        'extra':    ""}  
    }

# Segmentation fault
network = {
    0: {'model':    'models/pomdp/network/network2K-20_T-8_extended-simple_full.drn',      
        'formula':  'R{"dropped_packets"}=? [F "goal"]',
        'extra':    "--goal_label \"{'goal'}\""}  
    }

f = 'P=? ["notbad" U "goal"]'
e = "--goal_label \"{'goal','notbad'}\""

drone = {
    0: {'model':    'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn',      
        'formula':  f,
        'extra':    e},
    1: {'model':    'models/pomdp/drone/pomdp_drone_4-2-mem5-simple.drn',      
        'formula':  f,
        'extra':    e}  
    }

f = 'P=? [F "goal"]'
e = "--default_valuation 0.01 --goal_label \"{'goal'}\""  

satellite = {
    0: {'model':    'models/satellite/pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06_sparse_full.drn',      
        'formula':  f,
        'extra':    e},
    1: {'model':    'models/satellite/pomdp_prob_36_sat_065_dist_1_obs_diff_orb_len_40.drn',      
        'formula':  f,
        'extra':    e}  
    }



suites = [
    brp, crowds, nand, virus, wlan, csma, coin,
    maze, drone #, satellite
    ]

# assert False

# %%

for suite in suites:
    for i,exp in suite.items():
        run_experiment(exp, prefix_pmc, output, suffix)
       
# %%
       
for suite in suites:
    for i,exp in suite.items():
        run_experiment(exp, prefix_prmc, output, suffix)
        
        