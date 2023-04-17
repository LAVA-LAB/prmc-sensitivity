#!/bin/bash
cd ..;
echo -e "CREATE DATA FOR MOTIVATING EXAMPLE IN PAPER...";
#
python3 run_pmc.py --model 'models/slipgrid/dummy.nm' --parameters 'models/slipgrid/dummy.json' --formula 'Rmin=? [F "goal"]' --validate_delta 0.001 --output_folder 'output/motivating_example/' --num_deriv 4;
#
python3 run_pmc.py --model 'models/slipgrid/dummy.nm' --parameters 'models/slipgrid/dummy_mle.json' --formula 'Rmin=? [F "goal"]' --validate_delta 0.001 --output_folder 'output/motivating_example/' --num_deriv 4;
#
python3 run_prmc.py --model 'models/slipgrid/dummy.nm' --parameters 'models/slipgrid/dummy_mle.json' --formula 'Rmin=? [F "goal"]' --validate_delta 0.001 --output_folder 'output/motivating_example/' --num_deriv 4;
