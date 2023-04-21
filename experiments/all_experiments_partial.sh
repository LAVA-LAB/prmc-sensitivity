#!/bin/bash
echo -e "START FULL SET OF EXPERIMENTS...";

echo -e "\nStart grid world experiments...\n\n";
bash grid_world_partial.sh;

echo -e "\nStart other benchmarks...\n\n";
bash benchmarks_partial.sh;

echo -e "\nCreate data files for scatter plots...\n\n";
python3 create_scatter_data.py --files "['output/slipgrid_partial_table.csv', 'output/benchmarks_partial_table.csv']"

echo -e "\nStart learning experiments...\n\n";
cd ..;
python3 run_learning.py --instance gridworld --model models/slipgrid_learning/pmc_size=20_params=100.drn --parameters models/slipgrid_learning/pmc_size=20_params=100_mle.json --formula 'Rmin=? [F "goal"]' --output_folder 'output/learning/' --num_deriv 1 --robust_bound 'upper' --uncertainty_model 'Hoeffding' --true_param_file models/slipgrid_learning/pmc_size=20_params=100.json --learning_iterations 1 --learning_steps 100 --learning_samples_per_step 25;
python3 run_learning.py --instance drone --model models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn --formula 'P=? ["notbad" U "goal"]' --output_folder 'output/learning/' --num_deriv 1 --robust_bound 'upper' --uncertainty_model 'Hoeffding' --goal_label "{'goal','notbad'}" --default_sample_size 100 --learning_iterations 1 --learning_steps 100 --learning_samples_per_step 250;