from scipy import sparse

def define_sparse_LHS(model, point):

    subpoint = {}    
    
    row = []
    col = []
    val = []
    
    for state in model.states:
        for action in state.actions:
            for transition in action.transitions:
                
                ID = (state.id, action.id, transition.column)
                
                # Gather variables related to this transition
                var = transition.value().gather_variables()
                
                # Get valuation for only those parameters
                subpoint[ID] = {v: point[v] for v in var}
                
                if not model.is_sink_state(state.id):
                
                    value = float(transition.value().evaluate(subpoint[ID]))
                    if value != 0:
                
                        # Add to sparse matrix
                        row += [state.id]
                        col += [transition.column]
                        val += [value]
        
    # Create sparse matrix for left-hand side
    J = sparse.identity(len(model.states)) - sparse.csc_matrix((val, (row, col)), shape=(len(model.states), len(model.states)))
    
    return J, subpoint
    
def define_sparse_RHS(model, parameters, params2states, sols, subpoint):
    
    row = []
    col = []
    val = []
    
    for q,p in enumerate(parameters):
        
        for state in params2states[p]:
            
            # If the value in this state is zero, we can skip it anyways
            if sols[state.id] != 0:        
                for action in state.actions:
                    cur_val = 0
                    
                    for transition in action.transitions:
                        
                        ID = (state.id, action.id, transition.column)
                        
                        value = float(transition.value().derive(p).evaluate(subpoint[ID]))
                        
                        cur_val += value * sols[transition.column]
                        
                    if cur_val != 0:
                        row += [state.id]
                        col += [q]
                        val += [cur_val]
                 
    # Create sparse matrix for right-hand side
    Ju = -sparse.csc_matrix((val, (row, col)), shape=(len(model.states), len(parameters)))
    
    return Ju