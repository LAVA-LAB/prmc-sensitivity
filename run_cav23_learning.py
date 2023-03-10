from core.classes import PMC

from core.pmc_functions import pmc_load_instantiation, pmc_instantiate, assert_probabilities
from core.verify_pmc import pmc_verify, pmc_get_reward
from core.export import export_json, timer
from core.prmc_functions import pmc2prmc, prmc_verify, prmc_derivative_LP, prmc_validate_derivative
from core.parser import parse_main

from pathlib import Path
from tabulate import tabulate
from datetime import datetime

from core.learning.classes import learner

import os
import numpy as np
import pandas as pd

# Parse arguments
args = parse_main()
args.no_gradient_validation = True

SEEDS = np.arange(1)
MAX_STEPS = 10000
SAMPLES_PER_STEP = 25

# %%

# Load PRISM model with STORM
args.root_dir = os.path.dirname(os.path.abspath(__file__))

preset = 2

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
    args.goal_label = {'goal','notbad'}
    
    args.default_sample_size = 100
        
    args.beta_penalty = 0
    args.uncertainty_model = "Hoeffding"
    
    # args.true_param_file = 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.json'

model_path = Path(args.root_dir, args.model)
param_path = Path(args.root_dir, args.parameters) if args.parameters else False
true_param_path = Path(args.root_dir, args.true_param_file) if args.true_param_file else False

T = timer()

# %%

pmc = PMC(model_path = model_path, args = args)

args.uncertainty_model = "Hoeffding"
args.robust_probabilities = np.full(pmc.model.nr_states, True)
args.robust_dependencies = 'parameter' # Can be 'none' or 'parameter'

### pMC execution    
if true_param_path:
    # Load parameter valuation
    inst_true = pmc_load_instantiation(pmc, true_param_path, args.default_valuation)
    
else:
    # Create parameter valuation
    inst_true = {'valuation': {}}
    
    # Create parameter valuations on the spot
    for v in pmc.parameters:
        inst_true['valuation'][v.name] = args.default_valuation
        


# Compute True solution

# Define instantiated pMC based on parameter valuation
instantiated_model, inst_true['point'] = pmc_instantiate(pmc, inst_true['valuation'], T)
assert_probabilities(instantiated_model)

pmc.reward = pmc_get_reward(pmc, instantiated_model, args)

print('\n',instantiated_model,'\n')

# Verify true pMC
solution_true, J, Ju = pmc_verify(instantiated_model, pmc, inst_true['point'], T)

print('Optimal solution under the true parameter values: {:.3f}'.format(solution_true))

# TODO pmc does not do anything with a policy? Or implicitly?

# assert False



# %%

DFs = {}
DFs_stats = {}

modes = ['derivative','expVisits_sampling','expVisits','samples','random']

for mode in ['derivative', 'expVisits_sampling']: #modes[0:2]:
    
    DFs[mode] = pd.DataFrame()
    
    for q, seed in enumerate(SEEDS):
        print('>>> Start iteration {} <<<'.format(seed))
        
        if param_path:
            inst = pmc_load_instantiation(pmc, param_path, args.default_valuation)
            
        else:
            
            inst = {'valuation': {}, 'sample_size': {}}
            
            # Set seed
            np.random.seed(0)
            
            # Create parameter valuations on the spot
            for v in pmc.parameters:
                
                # Sample MLE value
                p = inst_true['valuation'][v.name]
                N = args.default_sample_size
                delta = 1e-4
                MLE = np.random.binomial(N, p) / N
                
                # Store
                inst['valuation'][v.name] = max(min(MLE , 1-delta), delta)
                inst['sample_size'][v.name] = args.default_sample_size
        
        # Define instantiated pMC based on parameter 
        instantiated_model, inst['point'] = pmc_instantiate(pmc, inst['valuation'], T)
        assert_probabilities(instantiated_model)

        pmc.reward = pmc_get_reward(pmc, instantiated_model, args)
        
        # model.reward += perturb
        
        # Define learner object
        L = learner(pmc, inst, SAMPLES_PER_STEP, seed, args, mode)
        
        for i in range(MAX_STEPS):
            print('----------------\nMethod {}, Iteration {}, Step {}\n----------------'.format(mode, q, i))
            
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