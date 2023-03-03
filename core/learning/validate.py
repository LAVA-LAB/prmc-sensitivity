import numpy as np
from core.main_prmc import pmc2prmc
from core.verify_prmc import verify_prmc

def validate(solution, validate_pars, args, pmc, inst):

    empirical_deriv = np.zeros(len(validate_pars))

    for q,x in enumerate(validate_pars):
        
        inst['sample_size'][x.name] += args.validate_delta
        
        args.beta_penalty = 0
        
        prmc_val = pmc2prmc(pmc.model, pmc.parameters, pmc.scheduler_prob, inst['point'], inst['sample_size'], args, verbose=False)
        
        CVX_GRB_val = verify_prmc(prmc_val, pmc.reward, args.robust_bound, verbose = False)
        
        # CVX_GRB_val.cvx.Params.NumericFocus = 3
        # CVX_GRB_val.cvx.Params.ScaleFlag = 1
        
        CVX_GRB_val.solve(store_initial = True)
        
        solution_new = CVX_GRB_val.x_tilde[pmc.sI['s']] @ pmc.sI['p']
        
        inst['sample_size'][x.name] -= args.validate_delta
        
        # Compute derivative
        empirical_deriv[q] = (solution_new-solution) / args.validate_delta
        
    return empirical_deriv