#!/bin/bash
cd ..;
echo -e "START GRID WORLD EXPERIMENTS...";
python3 run_pmc.py --instance "grid(5,20)" --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=5_params=20_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=5_params=20_seed=0.json' --formula 'Rmin=? [F "goal"]' --pMC_engine 'spsolve' --validate_delta 1e-6 --output_folder 'output/slipgrid/' --num_deriv 10 --explicit_baseline;
#
python3 create_table.py --folder 'output/slipgrid/' --table_name 'tables/slipgrid_table'