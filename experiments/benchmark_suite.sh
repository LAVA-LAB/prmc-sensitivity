#!/bin/bash
cd ..;
echo -e "START BENCHMARK SUITE...";
# Typical pdtmc benchmarks
python3 run_cav23.py --model 'models/pdtmc/parametric_die.pm' --formula 'P=? [ F s=7 & d=3 ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --goal_label '((s = 7) & (d = 3))' --verbose;
python3 run_cav23.py --model 'models/pdtmc/brp16_2.pm' --formula 'P=? [ F s=5 ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --goal_label '(s = 5)';
python3 run_cav23.py --model 'models/pdtmc/brp64_3.pm' --formula 'P=? [ F s=5 ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --goal_label '(s = 5)';
python3 run_cav23.py --model 'models/pdtmc/brp512_5.pm' --formula 'P=? [ F s=5 ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --goal_label '(s = 5)';
python3 run_cav23.py --model 'models/pdtmc/crowds3_5.pm' --formula 'P=? [F "observe0Greater1" ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --goal_label '(observe0Greater1)';
python3 run_cav23.py --model 'models/pdtmc/crowds6_5.pm' --formula 'P=? [F "observe0Greater1" ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --goal_label '(observe0Greater1)';
python3 run_cav23.py --model 'models/pdtmc/crowds10_5.pm' --formula 'P=? [F "observe0Greater1" ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --goal_label '(observe0Greater1)';
python3 run_cav23.py --model 'models/pdtmc/crowds20_10.pm' --formula 'P=? [F "observe0Greater1" ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --goal_label '(observe0Greater1)';
python3 run_cav23.py --model 'models/pdtmc/nand2_4.pm' --formula 'P=? [F "target" ]' --parameters 'models/pdtmc/nand.json' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --goal_label '(target)';
python3 run_cav23.py --model 'models/pdtmc/nand5_10.pm' --formula 'P=? [F "target" ]' --parameters 'models/pdtmc/nand.json' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --goal_label '(target)';
#
# STTT Drone
python3 run_cav23.py --model 'models/sttt-drone/drone_model.nm' --formula 'Pmax=? [F attarget ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --default_valuation 0.07692307692 --explicit_baseline;
#
# POMDP benchmarks
python3 run_cav23.py --model 'models/pomdp/maze/maze_simple_extended_m5.drn' --formula 'Rmin=? [F "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline; 
python3 run_cav23.py --model 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn' --formula 'P=? ["notbad" U "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --goal_label 'goal';
python3 run_cav23.py --model 'models/pomdp/drone/pomdp_drone_4-2-mem5-simple.drn' --formula 'P=? ["notbad" U "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --goal_label 'goal';
python3 run_cav23.py --model 'models/satellite/pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06_sparse_full.drn' --formula 'P=? [F "goal"]' --default_valuation 0.1 --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline --verbose;
python3 run_cav23.py --model 'models/satellite/pomdp_prob_36_sat_065_dist_1_obs_diff_orb_len_40.drn' --formula 'P=? [F "goal"]' --default_valuation 0.2 --validate_delta 1e-3 --output_folder 'output/benchmark_suite/' --explicit_baseline;
%
python3 parse_output.py --folder 'output/benchmark_suite/' --table_name 'tables/benchmark_suite'