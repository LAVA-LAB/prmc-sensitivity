import cvxpy as cp
from core.poly import poly

def verify_cvx(M, policy, verbose = True, solver = 'SCS'):
    
    if verbose:
        print('\nSolve LP for given formula...')
    
    # Define decision variables
    x = cp.Variable(len(M.states), nonneg=True)
    alpha = {}
    beta  = {}
    
    for s in M.states: 
        for a in s.actions:
            if a.robust:
                alpha[(s.id, a.id)] = cp.Variable(len(a.model.b))
                beta[(s.id, a.id)]  = cp.Variable()
    
    # Objective function
    objective = cp.Maximize( cp.sum([prob*x[s] for s,prob in M.sI.items()]) )
    
    cns = {}
    RHS = {}
    
    # Constraints per state (under the provided policy)
    for s in M.states:
        if verbose:
            print(' - Add constraints for state', s)
            
        # If not a terminal state
        if not s.terminal:
            
            if policy.deterministic:
                choice = policy.get_choice(s.id)
                action = choice.get_deterministic_choice()
                pol = {action: 1}
            
            RHS[s] = 0
                
            for a_id,prob in pol.items():
                
                a = s.actions_dict[a_id]
                
                if a.robust:
                    
                    # Add constraints for an uncertain probability distribution
                    b_alpha = [b.expr()*alph if isinstance(b, poly) else b*alph
                               for b,alph in zip(a.model.b, alpha[(s.id, a.id)])]
                    
                    RHS[s] -= prob * (cp.sum(b_alpha) + beta[(s.id, a.id)])
                    
                    # Add constraints on dual variables for each state-action pair
                    cns[(s.id, a.id)] = a.model.A.T @ alpha[(s.id, a.id)] + x[a.successors] + beta[(s.id, a.id)] == 0
                
                    cns[('ineq',s,a)] = alpha[(s.id, a.id)] >= 0
                    
                else:
                    
                    # Add constraints for a precise probability distribution
                    RHS[s] = prob * (x[a.model.states] @ a.model.probabilities)
                    
                    
                print( M.rewards.get_state_reward(s.id) )
                print( RHS[s] )
                cns[s] = x[s.id] == M.rewards.get_state_reward(s.id) + RHS[s]
            
        # If terminal state, reward equals instantaniously given value
        else:
            cns[s] = x[s.id] == M.rewards.get_state_reward(s.id)
            
    cvx_prob = cp.Problem(objective = objective, constraints = cns.values())
    
    if verbose:
        print('Is problem DCP?', cvx_prob.is_dcp(dpp=True))
    
    if solver == 'GUROBI':
        cvx_prob.solve(solver='GUROBI')
    else:
        cvx_prob.solve(solver='SCS', requires_grad=True, eps=1e-14, max_iters=10000, mode='dense')
        cvx_prob.backward()
    
    if verbose:
        print('Status:', cvx_prob.status)
    
    return cvx_prob, cns, x, alpha, beta
















def solve_LP(states, sI, edges, states_post, states_pre, states_nonterm, 
             reward, verbose=True):
    # Sparse implementation of reachability computation via LP
    
    if verbose:
        print('\nSolve LP for expected reward...')
    
    sp_x = cp.Variable(len(states), nonneg=True)
    sp_aLow = {(s,ss): cp.Variable() for (s,ss) in edges.keys()}
    sp_aUpp = {(s,ss): cp.Variable() for (s,ss) in edges.keys()}
    sp_beta = {s: cp.Variable() for s in states_nonterm}
    
    constraints = {}
    
    for s in states:
        if verbose:
            print(' - Add constraints for state', s)
    
        constraints[('nu',s)] = sp_x[s] == reward[s] \
                + cp.sum([sp_aLow[(s,ss)]*edges[(s,ss)][0].val() for ss in states_post[s]]) \
                - cp.sum([sp_aUpp[(s,ss)]*edges[(s,ss)][1].val() for ss in states_post[s]]) \
                - (sp_beta[s] if s in states_nonterm else 0)
        
    for (s,ss),e in edges.items():
        if verbose:
            print(' - Add edge from',s,'to',ss)
    
        constraints[('la_low',s,ss)] = sp_aLow[(s,ss)] >= 0 
        constraints[('la_upp',s,ss)] = sp_aUpp[(s,ss)] >= 0
            
        constraints[('mu',s,ss)] = sp_aUpp[(s,ss)] - sp_aLow[(s,ss)] + sp_x[ss] + sp_beta[s] == 0
        
    # Concatenate all constraints into a list
    sp_prob = cp.Problem(objective = cp.Maximize(sp_x[sI]), constraints = constraints.values())
    
    if verbose:
        print('Is problem DCP?', sp_prob.is_dcp(dpp=True))
    
    sp_prob.solve(requires_grad=True, eps=1e-14, max_iters=10000, mode='dense')
    sp_prob.backward()
    
    if verbose:
        print('Status:', sp_prob.status)
    
    return constraints, sp_x, sp_aLow, sp_aUpp

def sensitivity_LP(states, edges, states_post, states_pre, states_nonterm,
                   constraints, aLow, aUpp, verbose=False):
    # Perform sensitivity analysis
    
    if verbose:
        print('\nDefine system of equations for sensitivity analysis...')
    
    Dth_x = cp.Variable(len(states))
    Dth_aLow = {(s,ss): cp.Variable() for (s,ss) in edges.keys()}
    Dth_aUpp = {(s,ss): cp.Variable() for (s,ss) in edges.keys()}
    Dth_beta = {s: cp.Variable() for s in states_nonterm}
    
    Dth_nu = cp.Variable(len(states))
    Dth_mu = {(s,ss): cp.Variable() for (s,ss) in edges.keys()}
    
    GC = []
    
    # Differentiate lower bound from 0 to 1
    X = {}
    Y = {}
    Z = {}
    
    # Enumerate over states
    for s in states:
        if verbose:
            print(' - Add constraints for state', s)
        
        if s in states_nonterm:
            beta_sub = Dth_beta[s]
            
            # 6
            GC += [Dth_nu[s] + cp.sum([Dth_mu[(s,ss)] for ss in states_post[s]]) == 0]
        else:
            beta_sub = 0
            
        # 1
        GC += [Dth_nu[s] + cp.sum([Dth_mu[(ss,s)] for ss in states_pre[s]]) == 0]
        
        # 2
        X[(s)] = cp.Parameter()
        GC += [Dth_x[s] + cp.sum([- edges[(s,ss)][0].val()*Dth_aLow[(s,ss)]
                                  + edges[(s,ss)][1].val()*Dth_aUpp[(s,ss)] 
                                  for ss in states_post[s]])
                        + beta_sub == -X[(s)]]
        
    # Enumerate over edges
    for (s,ss),e in edges.items():
        if verbose:
            print(' - Add edge from',s,'to',ss)
        
        Y[(s,ss)] = cp.Parameter()
        Z[(s,ss)] = cp.Parameter()
        
        GC += [ 
            # 3 + 4
            constraints[('la_low',s,ss)].dual_value / aLow[(s,ss)].value * Dth_aLow[(s,ss)] == 
                ( e[0].val()*Dth_nu[s] + Dth_mu[(s,ss)] - Y[(s,ss)] ),
            constraints[('la_upp',s,ss)].dual_value / aUpp[(s,ss)].value * Dth_aUpp[(s,ss)] == 
                ( - e[1].val()*Dth_nu[s] - Dth_mu[(s,ss)] + Z[(s,ss)] ),
            # 5
            Dth_x[ss] - Dth_aLow[(s,ss)] + Dth_aUpp[(s,ss)] + Dth_beta[s] == 0
            ]
        
    Dth_prob = cp.Problem(cp.Maximize(0), constraints = GC)
        
    return Dth_prob, Dth_x, X, Y, Z