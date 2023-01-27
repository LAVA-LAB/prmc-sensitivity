#!/bin/bash
cd ..;
echo -e "CREATE DATA FOR MOTIVATING EXAMPLE IN PAPER...";
python3 run_pmc.py --model '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/dummy.nm' --parameters '/home/thom/documents/sensitivity-prmdps/prmdp-sensitivity-git/models/slipgrid/dummy.json' --formula 'Rmin=? [F "goal"]' --validate_delta 0.001 --output_folder 'output/motivating_example/';
