from core.main_prmc import pmc2prmc
from core.sensitivity import gradient, solve_cvx_gurobi
from core.verify_prmc import cvx_verification_gurobi

from core.learning.exp_visits import parameter_importance_exp_visits
from core.learning.validate import validate

from gurobipy import GRB

import pandas as pd
import numpy as np
from datetime import datetime
import random
import time

class learner:
    
    def __init__(self, pmc, inst, samples_per_step, seed, args, mode):
        
        self.pmc = pmc
        self.inst = inst
        
        self.SAMPLES_PER_STEP = samples_per_step
        self.args = args
        self.args.instance = 'seed_{}'.format(seed)
        
        self._set_sampler(mode)
        
        np.random.seed(seed) 
        random.seed(seed)
        
        self.solution_list = []
    
        # Define prMC
        self.prmc = pmc2prmc(self.pmc.model, self.pmc.parameters, self.inst['point'], self.inst['sample_size'], self.args, verbose = self.args.verbose)
        
        self.CVX = cvx_verification_gurobi(self.prmc, self.pmc.reward, self.args.robust_bound, verbose = self.args.verbose)
        self.CVX.cvx.tune()
        try:
            self.CVX.cvx.getTuneResult(0)
        except:
            print('Ecception: could not set tuning results')
        
        self.CVX.cvx.Params.NumericFocus = 3
        self.CVX.cvx.Params.ScaleFlag = 1
        
        if self.opposite:
            if self.args.robust_bound == 'lower':
                bound = 'upper'
            else:
                bound = 'lower'
            
            self.CVX_opp = cvx_verification_gurobi(self.prmc, self.pmc.reward, bound, verbose = self.args.verbose)
            self.CVX_opp.cvx.tune()
            try:
                self.CVX_opp.cvx.getTuneResult(0)
            except:
                print('Ecception: could not set tuning results')
        
        
    def _set_sampler(self, mode):
        
        if mode == 'random':
            self.sample_method = sample_uniform
            self.opposite = False
        elif mode == 'samples':
            self.sample_method = sample_lowest_count
            self.opposite = False
        elif mode == 'expVisits':
            self.sample_method = sample_importance
            self.opposite = True
        elif mode == 'expVisits_sampling':
            self.sample_method = sample_importance_proportional
            self.opposite = True
        elif mode == 'derivative':
            self.sample_method = sample_derivative
            self.opposite = False
        else:
            print('ERROR: unknown mode')
            assert False
        
        
    def solve_step(self):
        
        start_time = time.time()
        self.CVX.solve(store_initial = True, verbose=self.args.verbose)
        print(time.time() - start_time)
        
        self.solution_current = self.CVX.x_tilde[self.prmc.sI['s']] @ self.prmc.sI['p']
        self.solution_list += [np.round(self.solution_current, 2)]
        
        print('Range of solutions: [{}, {}]'.format(np.min(self.CVX.x_tilde), np.max(self.CVX.x_tilde)))
        print('Solution in initial state: {}\n'.format(self.solution_current))
        
        SLACK = self.CVX.get_active_constraints(self.prmc, verbose=False)
        
        if self.opposite:
            self.CVX_opp.solve(store_initial = True, verbose=self.args.verbose)
            SLACK = self.CVX_opp.get_active_constraints(self.prmc, verbose=True)
        
        
        
    def sample(self, params, true_valuation):
        
        for q,var in enumerate(params):
        
            if type(true_valuation) == dict:
                true_prob = true_valuation[var.name]
            else:
                true_prob = self.args.default_valuation
            
            samples = np.random.binomial(self.SAMPLES_PER_STEP, true_prob)
            
            old_sample_mean = self.inst['valuation'][var.name]
            new_sample_mean = (self.inst['valuation'][var.name] * self.inst['sample_size'][var.name] + samples) / (self.inst['sample_size'][var.name] + self.SAMPLES_PER_STEP)
            
            self.inst['valuation'][var.name] = new_sample_mean
            self.inst['sample_size'][var.name] += self.SAMPLES_PER_STEP
            
            print('\n>> Drawn {} more samples for parameter {} ({} positives)'.format(self.SAMPLES_PER_STEP, var, samples))
            print('>> MLE is now: {:.3f} (difference: {:.3f})'.format(new_sample_mean, new_sample_mean - old_sample_mean))
            print('>> Total number of samples is now: {}\n'.format(self.inst['sample_size'][var.name]))
        
        
        
    def update(self, params):
        
        ##### UPDATE PARAMETER POINT
        _, self.inst['point'] = self.pmc.instantiate(self.inst['valuation'])
        
        UPDATE = False
        
        if UPDATE:
        
            for var in params:
                # Update sample size
                self.prmc.parameters[var].value = self.inst['sample_size'][var.name]
                
                # Update mean
                self.prmc.update_distribution(var, self.inst)
                
                self.CVX.update_parameter(self.prmc, var)
                
                if self.opposite:
                    self.CVX_opp.update_parameter(self.prmc, var)
                
                # Update ordering over robust constraints
                self.prmc.set_robust_constraints()
            
        else:

            self.prmc = pmc2prmc(self.pmc.model, self.pmc.parameters, self.inst['point'], self.inst['sample_size'], self.args, verbose = self.args.verbose)
            self.CVX = cvx_verification_gurobi(self.prmc, self.pmc.reward, self.args.robust_bound, verbose = self.args.verbose)   
            
        
        
    def validate_derivatives(self, obj):
        
        print('\nValidation by perturbing parameters by +{}'.format(self.args.validate_delta))
        
        empirical_der = validate(self.solution_current, self.pmc.parameters, self.args, self.pmc, self.inst)
        relative_diff = (empirical_der/obj[0])-1
        
        for q,x in enumerate(self.pmc.parameters):
            print('- Parameter {}, val: {:.3f}, LP: {:.3f}, diff: {:.3f}'.format(
                    x,  empirical_der[q], obj[0], relative_diff[q]))
            
        min_deriv_val = self.pmc.parameters[ np.argmin(empirical_der) ]
        assert min_deriv_val in self.PAR
        assert np.isclose( np.min(empirical_der), obj[0] )
        
    
def sample_uniform(L):
    """
    Sample uniformly (randomly)
    """
    
    print('Sample uniformly...')
    
    idx = np.array([np.random.randint(0, len(L.prmc.parameters_pmc))])            
    PAR = L.prmc.parameters_pmc[idx]
    
    return PAR

def sample_lowest_count(L):
    """
    Always sample from parameter with lowest sample size so far.
    """
    
    print('Sample biggest interval...')
    
    # Get parameter with minimum number of samples so far
    par_samples = {}
    for key in L.prmc.parameters_pmc:
        par_samples[key] = L.inst['sample_size'][key.name]
        
    PAR = [min(par_samples, key=par_samples.get)]

    return PAR

def sample_importance(L):
    """
    Sample greedily based on importance factor
    """

    print('Sample based on importance factor (expVisits * intervalWidth...')

    importance, dtmc = parameter_importance_exp_visits(L.pmc, L.prmc, L.inst, L.CVX_opp)
    PAR = [max(importance, key=importance.get)]
    
    return PAR

def sample_importance_proportional(L):
    """
    Sample proportional to importance factors
    """
    
    print('Weighted sampling based on importance factor (expVisits * intervalWidth...')
    
    importance, dtmc = parameter_importance_exp_visits(L.pmc, L.prmc, L.inst, L.CVX_opp)
    
    keys    = list(importance.keys())
    weights = np.array(list(importance.values()))
    weights_norm = weights / sum(weights)
    
    PAR = [np.random.choice(keys, p=weights_norm)]
    
    return PAR

def sample_derivative(L):
    """
    Sample greedily based on the highest/lowest derivative
    """
    
    print('Sample based on biggest absolute derivative...')
    
    # Create object for computing gradients
    G = gradient(L.prmc, L.args.robust_bound)
    
    # Update gradient object with current solution
    G.update(L.prmc, L.CVX, mode='remove_dual')
    
    assert G.J.shape[0] == G.J.shape[1]
    
    if L.args.robust_bound == 'lower':
        direction = GRB.MAXIMIZE
    else:
        direction = GRB.MINIMIZE
        
    idx, obj = solve_cvx_gurobi(G.J, G.Ju, L.prmc.sI, L.args.num_deriv,
                                direction=direction, verbose=L.args.verbose, method=5)

    PAR = [G.col2param[v] for v in idx]
    
    if L.args.validate:        
        L.validate_derivatives(obj)
    
    return PAR