#!/bin/bash
cd ..;
echo -e "START EXPERTIMENTS...\n";
python3 run.py --model 'models/mdp/slipgrid.nm' --formula 'Rmin=? [F "goal"]' --terminal_label 'goal';
python3 run.py --model 'models/dtmc/dummy.nm' --terminal_label 'done';