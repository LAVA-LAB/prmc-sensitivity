# %run "~/documents/sensitivity-prmdps/prmdp-sensitivity-git/run.py"

from core.main_pmc import run_pmc, export_pmc
from core.main_prmc import run_prmc

from core.parse_inputs import parse_inputs
import os
from pathlib import Path
from tabulate import tabulate
from datetime import datetime

# Parse arguments
args = parse_inputs()

# Load PRISM model with STORM
args.root_dir = os.path.dirname(os.path.abspath(__file__))

args.model = 'models/slipgrid/dummy.nm'
args.parameters = 'models/slipgrid/dummy_mle.json'

args.formula = 'Rmin=? [F \"goal\"]'
args.num_deriv = 1
args.validate_delta = 1e-6

### Parse model

current_time = datetime.now().strftime("%H:%M:%S")
print('Program started at {}'.format(current_time))
print('\n',tabulate(vars(args).items(), headers=["Argument", "Value"]),'\n')

model_path = Path(args.root_dir, args.model)
param_path = Path(args.root_dir, args.parameters) if args.parameters else False

pmc, T, inst, solution, deriv = run_pmc(args, model_path, param_path)

export_pmc(args, pmc, T, inst, solution, deriv)

current_time = datetime.now().strftime("%H:%M:%S")
print('\nProgram ended at {}\n'.format(current_time))


run_prmc(pmc, args, inst)