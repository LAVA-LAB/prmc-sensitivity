#!/bin/bash
cd ..;
echo -e "START GRID WORLD EXPERIMENTS...";
python3 run_cav23.py --instance "grid(800,100)" --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/fix_pmc_size=800_params=100_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/fix_pmc_size=800_params=100_seed=0_mle.json' --formula 'Rmin=? [F "goal"]' --pMC_engine 'spsolve' --validate_delta -0.001 --output_folder 'output/slipgrid/' --num_deriv 1 --explicit_baseline --robust_bound 'lower' --no_prMC --scale_reward;
python3 run_cav23.py --instance "grid(800,100)" --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/double_pmc_size=800_params=100_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/double_pmc_size=800_params=100_seed=0_mle.json' --formula 'Rmin=? [F "goal"]' --pMC_engine 'spsolve' --validate_delta -0.001 --output_folder 'output/slipgrid/' --num_deriv 1 --explicit_baseline --robust_bound 'lower' --no_prMC --scale_reward;
#
python3 create_table.py --folder 'output/slipgrid/' --table_name 'tables/slipgrid_table'