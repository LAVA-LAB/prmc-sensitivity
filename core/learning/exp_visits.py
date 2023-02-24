import numpy as np
import stormpy
from core.commons import valuate
from scipy.linalg import solve

def dtmc_active_bounds(prmc, instantiated_model, active_constraints):
    
    for s in instantiated_model.states:
        
        for a in s.actions:
            
            if instantiated_model.is_sink_state(s):
                continue
                
            a_prmc = prmc.states_dict[s.id].actions_dict[a.id]
            
            # Get polytope
            A = a_prmc.model.A
            b = valuate(a_prmc.model.b)
            
            # Retrieve actual constraints
            active = active_constraints[(s.id, a.id)]
            
            # Add constraint for probabilities summing to one
            A_one = np.ones((1,A.shape[1]))
            b_one = np.array([1])
            
            # Define full eq. sys for active point
            A_active = np.concatenate(( A[active, :], A_one ))
            b_active = np.concatenate(( b[active], b_one ))
            
            # print(A_active, b_active)
            
            # Get worst-case point
            worstcase_point = solve(A_active, b_active)
            
            if np.sum(worstcase_point) != 1:
                print('Warning: normalize worstcase point (by {:.3f}) to one for ({},{})'.format(np.sum(worstcase_point), s.id, a.id))
                worstcase_point /= np.sum(worstcase_point)
            
            # Update probabilities in dtmc with worst-case point
            for i,t in enumerate(a.transitions):
                # check if successor states agree
                assert t.column == a_prmc.successors[i]
                
                t.set_value(worstcase_point[i])
                
            # print(s,a,'-',worstcase_point)
            # print(b)
            # print(A)
            
            # assert False
            
    return instantiated_model


def parameter_importance_exp_visits(pmc, prmc, inst, CVX_GRB):
    
    instantiated_model, _ = pmc.instantiate(inst['valuation'])
    
    dtmc = dtmc_active_bounds(prmc, instantiated_model, CVX_GRB.active_constraints)
    
    expected_number_of_visits = stormpy.compute_expected_number_of_visits(stormpy.Environment(),
                                                                          dtmc)
    
    importance = {}

    for s in instantiated_model.states:
        for a in s.actions:
            
            if instantiated_model.is_sink_state(s):
                continue
            
            for v in prmc.stateAction2param[(s.id, a.id)]:
                
                # print('Visits s={}: {}'.format(s.id, expected_number_of_visits.at(s.id)))
                
                #imp = expected_number_of_visits.at(s.id) / inst['sample_size'][v.name]
                imp = expected_number_of_visits.at(s.id) * np.sqrt(1/inst['sample_size'][v.name])
                
                if v in importance:
                    importance[v] += imp
                else:
                    importance[v] = imp
                    
    # print(importance)
           
    return importance, dtmc