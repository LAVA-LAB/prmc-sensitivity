import stormpy
import stormpy.examples
import stormpy.examples.files

def load_prism_model(path):
    
    print('Load PRISM model with STORM...')
    
    program = stormpy.parse_prism_program(path)
    model = stormpy.build_model(program)
    

    return model