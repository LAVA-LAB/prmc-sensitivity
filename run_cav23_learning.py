from core.main_pmc import run_pmc, pmc_instantiate, pmc_set_reward
from core.classes import PMC
from core.parser import parse_main
import json

import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import datetime

from core.learning.classes import learner

# Parse arguments
args = parse_main()
args.validate = False

SEEDS = np.arange(2)
MAX_STEPS = 10
SAMPLES_PER_STEP = 25

# %%

# Load PRISM model with STORM
args.root_dir = os.path.dirname(os.path.abspath(__file__))

preset = 0

if preset == 0:

    N = 20
    V = 100   
    seed = 0

    args.model = 'models/slipgrid_learning/double_pmc_size={}_params={}_seed={}.drn'.format(N,V,seed)
    args.parameters = 'models/slipgrid_learning/double_pmc_size={}_params={}_seed={}_mle.json'.format(N,V,seed)
    args.formula = 'Rmin=? [F "goal"]'
    args.pMC_engine = 'spsolve'
    args.output_folder = 'output/learning/'
    args.num_deriv = 1
    args.robust_bound = 'upper'
    
    args.beta_penalty = 0
    args.uncertainty_model = "Hoeffding"
    
    args.true_param_file = 'models/slipgrid_learning/double_pmc_size={}_params={}_seed={}.json'.format(N,V,seed)

elif preset == 1:
    
    args.model = 'models/pmdp/virus/virus.pm'
    args.parameters = 'models/pmdp/virus/virus_mle.json'
    args.formula = 'R{"attacks"}max=? [F s11=2 ]'
    args.pMC_engine = 'storm'
    args.output_folder = 'output/learning/'
    args.num_deriv = 1
    args.robust_bound = 'upper'
    
    # args.beta_penalty = 0
    args.uncertainty_model = "Hoeffding"
    
    args.true_param_file = 'models/pmdp/virus/virus.json'
    
elif preset == 2:
    
    args.model = 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn'
    # args.parameters = 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple_mle.json'
    args.formula = 'P=? ["notbad" U "goal"]'
    args.pMC_engine = 'spsolve'
    args.output_folder = 'output/learning/'
    args.num_deriv = 1
    args.robust_bound = 'upper'
    args.goal_label = 'goal' 
    
    args.default_sample_size = 1000
    args.default_valuation = 0.5
    
    args.beta_penalty = 0
    args.uncertainty_model = "Hoeffding"
    
    # args.true_param_file = 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.json'

model_path = Path(args.root_dir, args.model)
param_path = Path(args.root_dir, args.parameters) if args.parameters else False
true_param_path = Path(args.root_dir, args.true_param_file) if args.true_param_file else False

# %%

model = PMC(model_path = model_path, args = args)

### pMC execution    
inst_true = {}
if true_param_path:
    # Load parameter valuation
    
    inst_true['valuation'], _ = model.load_instantiation(args = args, param_path = true_param_path)
    
else:
    # Create parameter valuation
    
    inst_true['valuation'] = {}
    
    # Create parameter valuations on the spot
    for v in model.parameters:
        inst_true['valuation'][v.name] = args.default_valuation
        
        
inst = {}
if param_path:
    inst['valuation'], inst['sample_size'] = model.load_instantiation(args = args, param_path = param_path)
    
else:
    
    inst['valuation'] = {}
    inst['sample_size'] = {}
    
    # Set seed
    np.random.seed(0)
    
    # Create parameter valuations on the spot
    for v in model.parameters:
        
        # Sample MLE value
        p = inst_true['valuation'][v.name]
        N = args.default_sample_size
        delta = 1e-4
        MLE = np.random.binomial(N, p) / N
        
        # Store
        inst['valuation'][v.name] = max(min(MLE , 1-delta), delta)
        inst['sample_size'][v.name] = args.default_sample_size

# assert False

# Compute True solution

# Define instantiated pMC based on parameter valuation
instantiated_model, T = pmc_instantiate(args, model, inst_true, verbose = args.verbose)
pmc_set_reward(model, args, inst_true)

# perturb = 1e-4*np.random.rand(len(model.reward))
# model.reward += perturb

# Verifying pMC
_, _, solution_true, _ = run_pmc(args, model, instantiated_model, inst_true, verbose = args.verbose, T = T)

print('Optimal solution under the true parameter values: {:.3f}'.format(solution_true))

# TODO pmc does not do anything with a policy? Or implicitly?

# assert False

# %%

DFs = {}
DFs_stats = {}

for mode in ['derivative','samples']:
    
    DFs[mode] = pd.DataFrame()
    
    for seed in SEEDS:
        print('>>> Start iteration {} <<<'.format(seed))
        
        # Define pMC
        
        # Define instantiated pMC based on parameter valuation
        instantiated_model, T = pmc_instantiate(args, model, inst, verbose = args.verbose)
        pmc_set_reward(model, args, inst)

        # model.reward += perturb

        # Verifying pMC
        pmc, _, _, _ = run_pmc(args, model, instantiated_model, inst, verbose = args.verbose, T = T)
        
        print('- Create arbitrary sample sizes')
        # inst['sample_size'] = {par.name: 1000 + np.random.rand()*10 for par in pmc.parameters}
        
        # Define learner object
        L = learner(pmc, inst, SAMPLES_PER_STEP, seed, args, mode)
        
        for i in range(MAX_STEPS):
            print('----\nStep {}\n----'.format(i))
            
            # Compute robust solution for current step
            L.solve_step()
            
            # Determine for which parameter to obtain additional samples
            PAR = L.sample_method(L)
            
            # Get additional samples
            L.sample(PAR, inst_true['valuation'])
            
            # Update learnined object
            L.update(PAR)
            
        DFs[mode] = pd.concat([DFs[mode], pd.Series(L.solution_list)], axis=1)
        
    current_time = datetime.now().strftime("%H:%M:%S")
    print('\nprMC code ended at {}\n'.format(current_time))
    print('=============================================')
    
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    DFs[mode].to_csv('output/learning_{}_{}.csv'.format(dt,mode), sep=';')    
        
    DFs_stats[mode] = pd.DataFrame({
        '{}_mean'.format(mode): DFs[mode].mean(axis=1),
        '{}_min'.format(mode): DFs[mode].min(axis=1),
        '{}_max'.format(mode): DFs[mode].max(axis=1)
        })
        
# %%
    
import matplotlib.pyplot as plt

dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

DF_stats = pd.concat(list(DFs_stats.values()), axis=1)

df_merged = pd.concat([df.mean(axis=1) for df in DFs.values()], axis=1)
df_merged.columns = list(DFs.keys())

df_merged.plot()

plt.axhline(y=solution_true, color='gray', linestyle='--')

DF_stats.to_csv('output/learning_{}_merged.csv'.format(dt), sep=';') 

plt.savefig('output/learning_curves_{}.png'.format(dt))
plt.savefig('output/learning_curves_{}.pdf'.format(dt))

plt.show()