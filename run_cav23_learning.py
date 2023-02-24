from core.main_pmc import run_pmc
from core.main_prmc import run_prmc
from core.parse_inputs import parse_inputs

from core.main_prmc import pmc2prmc
from core.sensitivity import gradient, solve_cvx_gurobi
from core.verify_prmc import cvx_verification_gurobi

from core.learning.exp_visits import parameter_importance_exp_visits
from core.learning.validate import validate

from gurobipy import GRB
import json

import pandas as pd
import os
import numpy as np
from pathlib import Path
from tabulate import tabulate
from datetime import datetime
import random

N = 20
V = 100
seed = 0

# Parse arguments
args = parse_inputs()

EXPORT = True
VALIDATE = False
UPDATE = True

SEEDS = np.arange(5)
MAX_STEPS = 1000
SAMPLES_PER_STEP = 25

# %%

# Load PRISM model with STORM
args.root_dir = os.path.dirname(os.path.abspath(__file__))

args.model = 'models/slipgrid_learning/double_pmc_size={}_params={}_seed={}.drn'.format(N,V,seed)
args.parameters = 'models/slipgrid_learning/double_pmc_size={}_params={}_seed={}_mle.json'.format(N,V,seed)
args.formula = 'Rmin=? [F "goal"]'
args.pMC_engine = 'spsolve'
args.output_folder = 'output/learning/'
args.num_deriv = 1
args.robust_bound = 'upper'
args.no_pMC = False

args.beta_penalty = 0
args.uncertainty_model = "Hoeffding"

### pMC execution

current_time = datetime.now().strftime("%H:%M:%S")
print('Program started at {}'.format(current_time))
print('\n',tabulate(vars(args).items(), headers=["Argument", "Value"]),'\n')

model_path = Path(args.root_dir, args.model)
param_path = Path(args.root_dir, args.parameters) if args.parameters else False

### prMC execution
true_param_file = 'models/slipgrid_learning/double_pmc_size={}_params={}_seed={}.json'.format(N,V,seed)
true_param_path = Path(args.root_dir, true_param_file)

# Read true parameter values
with open(str(true_param_path)) as json_file:
    true_valuation = json.load(json_file)
    
# Compute True solution
_, _, _, solution_true, _ = run_pmc(args, model_path, true_param_path, verbose = args.verbose)
print('Optimal solution under the true parameter values: {:.3f}'.format(solution_true))

# %%

DFs = {}
DFs_stats = {}

import time
for mode in ['derivative', 'expVisits', 'expVisits_sampling', 'samples', 'random']:
    
    if mode in ['expVisits', 'expVisits_sampling']:
        optim = True
    else:
        optim = False
    
    DFs[mode] = pd.DataFrame()
    
    for seed in SEEDS:
    
        print('>>> Start iteration {} <<<'.format(seed))
        
        args.instance = 'seed_{}'.format(seed)
        
        np.random.seed(seed) 
        random.seed(seed)
        
        pmc, _, inst, _, _ = run_pmc(args, model_path, param_path, verbose = args.verbose)
    
        SOL_LIST = []    
    
        prmc = pmc2prmc(pmc.model, pmc.parameters, inst['point'], inst['sample_size'], args, verbose = args.verbose)
        
        CVX_GRB = cvx_verification_gurobi(prmc, pmc.reward, args.robust_bound, verbose = args.verbose)
        CVX_GRB.cvx.tune()
        CVX_GRB.cvx.getTuneResult(0)
        
        if optim:
            CVX_GRB_opt = cvx_verification_gurobi(prmc, pmc.reward, 'lower', verbose = args.verbose)
            CVX_GRB_opt.cvx.tune()
            CVX_GRB_opt.cvx.getTuneResult(0)
        
        for i in range(MAX_STEPS):
        
            print('> Step {}'.format(i))    
        
            ##### PARAMETER IMPORTANCE    
            
            # CVX_GRB.cvx.Params.Method = 5
            # CVX_GRB.cvx.Params.Seed = 0   
            
            # CVX_GRB.cvx.Params.NumericFocus = 3
            # CVX_GRB.cvx.Params.ScaleFlag = 1
            
            start_time = time.time()
            CVX_GRB.solve(store_initial = True, verbose=args.verbose)
            print(time.time() - start_time)
            
            SOL = CVX_GRB.x_tilde[prmc.sI['s']] @ prmc.sI['p']
            SOL_LIST += [np.round(SOL, 2)]
            
            # print('Range of solutions: [{}, {}]'.format(np.min(CVX_GRB.x_tilde), np.max(CVX_GRB.x_tilde)))
            print('Solution in initial state: {}\n'.format(SOL))
            
            SLACK = CVX_GRB.check_complementary_slackness(prmc, verbose=True)
            
            if optim:
                CVX_GRB_opt.solve(store_initial = True, verbose=args.verbose)
                SLACK = CVX_GRB_opt.check_complementary_slackness(prmc, verbose=True)
            
            # Switch between random parameter choice or via derivative
            if mode == 'random':
    
                print('Sample uniformly...')            
    
                idx = np.array([np.random.randint(0, len(prmc.parameters_pmc))])            
                PAR = prmc.parameters_pmc[idx]
    
            elif mode == 'samples':
                print('Sample biggest interval...')
                
                # Get parameter with minimum number of samples so far
                par_samples = {}
                for key in prmc.parameters_pmc:
                    par_samples[key] = inst['sample_size'][key.name]
                    
                PAR = [min(par_samples, key=par_samples.get)]

                print('- Lower parameter count is {} for {}'.format(min(par_samples.values()), PAR[0].name))
                
            elif mode == 'expVisits':
                
                print('Sample based on importance factor (expVisits * intervalWidth...')
                
                importance, dtmc = parameter_importance_exp_visits(pmc, prmc, inst, CVX_GRB_opt)
                PAR = [max(importance, key=importance.get)]
                
            elif mode == 'expVisits_sampling':
                
                print('Weighted sampling based on importance factor (expVisits * intervalWidth...')
                
                importance, dtmc = parameter_importance_exp_visits(pmc, prmc, inst, CVX_GRB)
                
                keys    = list(importance.keys())
                weights = np.array(list(importance.values()))
                weights_norm = weights / sum(weights)
                
                PAR = [np.random.choice(keys, p=weights_norm)]
                
            elif mode =='derivative':            
    
                print('Sample based on biggest absolute derivative...')            
                
                # Create object for computing gradients
                G = gradient(prmc, args.robust_bound)
                
                # Update gradient object with current solution
                G.update(prmc, CVX_GRB, mode='reduce_dual')
                
                if args.robust_bound == 'lower':
                    direction = GRB.MAXIMIZE
                else:
                    direction = GRB.MINIMIZE
                    
                idx, obj = solve_cvx_gurobi(G.J, G.Ju, prmc.sI, args.num_deriv,
                                            direction=direction, verbose=args.verbose, method=5)
            
                PAR = [G.col2param[v] for v in idx]

                if VALIDATE:
                    
                    print('\nValidation by perturbing parameters by +{}'.format(args.validate_delta))
                    
                    empirical_der = validate(SOL, pmc.parameters, args, pmc, inst)
                    relative_diff = (empirical_der/obj[0])-1
                    
                    for q,x in enumerate(pmc.parameters):
                        print('- Parameter {}, val: {:.3f}, LP: {:.3f}, diff: {:.3f}'.format(
                                x,  empirical_der[q], obj[0], relative_diff[q]))
                        
                    min_deriv_val = pmc.parameters[ np.argmin(empirical_der) ]
                    assert min_deriv_val in PAR
                    assert np.isclose( np.min(empirical_der), obj[0] )
            
            else:
                print('ERROR: unknown mode')
                assert False
            
            ##### OBTAIN MORE SAMPLES
            for q,var in enumerate(PAR):
            
                true_prob = true_valuation[var.name]
                samples = np.random.binomial(SAMPLES_PER_STEP, true_prob)
                
                old_sample_mean = inst['valuation'][var.name]
                new_sample_mean = (inst['valuation'][var.name] * inst['sample_size'][var.name] + samples) / (inst['sample_size'][var.name] + SAMPLES_PER_STEP)
                
                inst['valuation'][var.name] = new_sample_mean
                inst['sample_size'][var.name] += SAMPLES_PER_STEP
                
                print('\n>> ({},{},{}) - Drawn {} more samples for parameter {} ({} positives)'.format(mode,seed, i, SAMPLES_PER_STEP, var, samples))
                print('>> MLE is now: {:.3f} (difference: {:.3f})'.format(new_sample_mean, new_sample_mean - old_sample_mean))
                print('>> Total number of samples is now: {}\n'.format(inst['sample_size'][var.name]))
                
            
            ##### UPDATE PARAMETER POINT
            _, inst['point'] = pmc.instantiate(inst['valuation'])
            
            if UPDATE:
                
                for var in PAR:
                    # Update sample size
                    prmc.parameters[var].value = inst['sample_size'][var.name]
                    
                    # Update mean
                    prmc.update_distribution(var, inst)
                    
                    CVX_GRB.update_parameter(prmc, var)
                    
                    if optim:
                        CVX_GRB_opt.update_parameter(prmc, var)
                    
                    # Update ordering over robust constraints
                    prmc.set_robust_constraints()
                    
            else:
                
                prmc = pmc2prmc(pmc.model, pmc.parameters, inst['point'], inst['sample_size'], args, verbose = args.verbose)
            
                CVX_GRB = cvx_verification_gurobi(prmc, pmc.reward, args.robust_bound, verbose = args.verbose)   
                
                if optim:
                    CVX_GRB_opt = cvx_verification_gurobi(prmc, pmc.reward, 'lower', verbose = args.verbose)                    
        
        DFs[mode] = pd.concat([DFs[mode], pd.Series(SOL_LIST)], axis=1)
        
    current_time = datetime.now().strftime("%H:%M:%S")
    print('\nprMC code ended at {}\n'.format(current_time))
    print('=============================================')
    
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if EXPORT:
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

if EXPORT:
    DF_stats.to_csv('output/learning_{}_merged.csv'.format(dt), sep=';') 
    
    plt.savefig('output/learning_curves_{}.png'.format(dt))
    plt.savefig('output/learning_curves_{}.pdf'.format(dt))

plt.show()