#!/bin/bash
cd ..;
echo -e "START GRID WORLD EXPERIMENTS...";
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=10_params=50_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=10_params=50_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=20_params=50_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=20_params=50_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=40_params=50_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=40_params=50_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=80_params=50_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=80_params=50_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=160_params=50_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=160_params=50_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=10_params=100_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=10_params=100_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=20_params=100_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=20_params=100_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=40_params=100_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=40_params=100_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=80_params=100_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=80_params=100_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=160_params=100_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=160_params=100_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=40_params=500_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=40_params=500_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=80_params=500_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=80_params=500_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=160_params=500_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=160_params=500_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=40_params=1000_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=40_params=1000_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=80_params=1000_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=80_params=1000_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=160_params=1000_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=160_params=1000_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=80_params=5000_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=80_params=5000_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=160_params=5000_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=160_params=5000_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=160_params=10000_seed=0.drn' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/pmc_size=160_params=10000_seed=0.json' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal'; 