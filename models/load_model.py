import stormpy
import stormpy.examples
import stormpy.examples.files

def load_prism_model(path, formula=None):
    
    print('Load PRISM model with STORM...')
    
    program = stormpy.parse_prism_program(path)
    model = stormpy.build_model(program)

    if model.is_nondeterministic_model:
        
        formulas = stormpy.parse_properties(formula, program)
        
        result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)
        policy = result.scheduler

    else:
        
        policy = None

    return model, policy