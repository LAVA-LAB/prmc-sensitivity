from models.prmdp import to_prmdp
from core.sensitivity import gradient, solve_cvx_gurobi
from core.cvx import cvx_verification_gurobi
import scipy.sparse as sparse
import time
from gurobipy import GRB

def run_prmc(pmc, args, inst):

    args.interval_confidence = 0.90
    args.uncertainty_model = "Hoeffding"
    args.robust_bound = 'upper'
    
    prmc = to_prmdp(pmc.model, pmc.parameters, inst['point'], inst['sample_size'], args)
    
    CVX_GRB = cvx_verification_gurobi(prmc, pmc.reward, args.robust_bound, verbose=True)
    CVX_GRB.solve(store_initial = True, verbose=True)
    
    sol_old = CVX_GRB.x_tilde[0]
    
    # %%
    
    # Check if complementary slackness is satisfied
    CVX_GRB.check_complementary_slackness(prmc)
    
    # Create object for computing gradients
    G = gradient(prmc, args.robust_bound)
    
    # Update gradient object with current solution
    G.update(prmc, CVX_GRB, mode='remove_dual')
    
    print('\n--------------------------------------------------------------')
    print('Solve LP using Gurobi...')
    start_time = time.time()
    
    if args.robust_bound == 'lower':
        direction = GRB.MAXIMIZE
    else:
        direction = GRB.MINIMIZE
        
    K, Deriv = solve_cvx_gurobi(G.J, G.Ju, pmc.sI, args.num_deriv,
                                direction=direction, verbose=False)
    
    grad = sparse.linalg.spsolve(G.J, -G.Ju)[pmc.sI['s']].T @ pmc.sI['p']
    print(grad)
    
    time_solve_LP = time.time() - start_time   
    print('- LP solved in: {:.3f} sec.'.format(time_solve_LP))
    print('--------------------------------------------------------------\n')    
    
    # %%
    
    delta = 1e-4
    inst['sample_size'][pmc.parameters[K[0]].name] += delta
    
    prmc = to_prmdp(pmc.model, pmc.parameters, inst['point'], inst['sample_size'], args)
    
    CVX_GRB = cvx_verification_gurobi(prmc, pmc.reward, args.robust_bound, verbose=True)
    CVX_GRB.solve(store_initial = True, verbose=True)
    
    sol_new = CVX_GRB.x_tilde[0]
    
    grad_emp = (sol_new-sol_old)/delta
    
    print('\n\nDerivatives:', Deriv)
    print('Grad_emp:', grad_emp)