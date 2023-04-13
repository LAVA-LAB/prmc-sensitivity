#!/bin/bash
cd ..;
echo -e "START BENCHMARK SUITE...";

# BENCHMARKS FOR brp
timeout 3600s python3 run_pmc.py --model 'models/pdtmc/brp16_2.pm' --formula 'P=? [ F s=5 ]' --default_valuation 0.9 --goal_label "{'(s = 5)'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pdtmc/brp16_2.pm' --formula 'P=? [ F s=5 ]' --default_valuation 0.9 --goal_label "{'(s = 5)'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pdtmc/brp16_2.pm' --formula 'P=? [ F s=5 ]' --default_valuation 0.9 --goal_label "{'(s = 5)'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_pmc.py --model 'models/pdtmc/brp16_2.pm' --formula 'P=? [ F s=5 ]' --default_valuation 0.9 --goal_label "{'(s = 5)'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pdtmc/brp16_2.pm' --formula 'P=? [ F s=5 ]' --default_valuation 0.9 --goal_label "{'(s = 5)'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pdtmc/brp16_2.pm' --formula 'P=? [ F s=5 ]' --default_valuation 0.9 --goal_label "{'(s = 5)'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;

# BENCHMARKS FOR crowds
timeout 3600s python3 run_pmc.py --model 'models/pdtmc/crowds3_5.pm' --formula 'P=? [F "observe0Greater1" ]' --goal_label "{'observe0Greater1'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pdtmc/crowds3_5.pm' --formula 'P=? [F "observe0Greater1" ]' --goal_label "{'observe0Greater1'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pdtmc/crowds3_5.pm' --formula 'P=? [F "observe0Greater1" ]' --goal_label "{'observe0Greater1'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_pmc.py --model 'models/pdtmc/crowds3_5.pm' --formula 'P=? [F "observe0Greater1" ]' --goal_label "{'observe0Greater1'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pdtmc/crowds3_5.pm' --formula 'P=? [F "observe0Greater1" ]' --goal_label "{'observe0Greater1'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pdtmc/crowds3_5.pm' --formula 'P=? [F "observe0Greater1" ]' --goal_label "{'observe0Greater1'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;

# BENCHMARKS FOR nand
timeout 3600s python3 run_pmc.py --model 'models/pdtmc/nand2_4.pm' --formula 'P=? [F "target" ]' --parameters 'models/pdtmc/nand.json' --goal_label "{'target'}" --validate_delta 1e-2 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pdtmc/nand2_4.pm' --formula 'P=? [F "target" ]' --parameters 'models/pdtmc/nand.json' --goal_label "{'target'}" --validate_delta 1e-2 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pdtmc/nand2_4.pm' --formula 'P=? [F "target" ]' --parameters 'models/pdtmc/nand.json' --goal_label "{'target'}" --validate_delta 1e-2 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_pmc.py --model 'models/pdtmc/nand2_4.pm' --formula 'P=? [F "target" ]' --parameters 'models/pdtmc/nand.json' --goal_label "{'target'}" --validate_delta 1e-2 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pdtmc/nand2_4.pm' --formula 'P=? [F "target" ]' --parameters 'models/pdtmc/nand.json' --goal_label "{'target'}" --validate_delta 1e-2 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pdtmc/nand2_4.pm' --formula 'P=? [F "target" ]' --parameters 'models/pdtmc/nand.json' --goal_label "{'target'}" --validate_delta 1e-2 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;

# BENCHMARKS FOR virus
timeout 3600s python3 run_pmc.py --model 'models/pmdp/virus/virus.pm' --formula 'R{"attacks"}max=? [F s11=2 ]' --default_valuation 0.1 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/virus/virus.pm' --formula 'R{"attacks"}max=? [F s11=2 ]' --default_valuation 0.1 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/virus/virus.pm' --formula 'R{"attacks"}max=? [F s11=2 ]' --default_valuation 0.1 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_pmc.py --model 'models/pmdp/virus/virus.pm' --formula 'R{"attacks"}max=? [F s11=2 ]' --default_valuation 0.1 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/virus/virus.pm' --formula 'R{"attacks"}max=? [F s11=2 ]' --default_valuation 0.1 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/virus/virus.pm' --formula 'R{"attacks"}max=? [F s11=2 ]' --default_valuation 0.1 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;

# BENCHMARKS FOR wlan
timeout 3600s python3 run_pmc.py --model 'models/pmdp/wlan/wlan0_param.nm' --formula 'R{"time"}max=? [ F s1=12 | s2=12 ]' --default_valuation 0.001 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/wlan/wlan0_param.nm' --formula 'R{"time"}max=? [ F s1=12 | s2=12 ]' --default_valuation 0.001 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/wlan/wlan0_param.nm' --formula 'R{"time"}max=? [ F s1=12 | s2=12 ]' --default_valuation 0.001 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_pmc.py --model 'models/pmdp/wlan/wlan0_param.nm' --formula 'R{"time"}max=? [ F s1=12 | s2=12 ]' --default_valuation 0.001 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/wlan/wlan0_param.nm' --formula 'R{"time"}max=? [ F s1=12 | s2=12 ]' --default_valuation 0.001 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/wlan/wlan0_param.nm' --formula 'R{"time"}max=? [ F s1=12 | s2=12 ]' --default_valuation 0.001 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;

# BENCHMARKS FOR csma
timeout 3600s python3 run_pmc.py --model 'models/pmdp/CSMA/csma2_4_param.nm' --formula 'R{"time"}max=? [ F "all_delivered" ]' --default_valuation 0.05 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/CSMA/csma2_4_param.nm' --formula 'R{"time"}max=? [ F "all_delivered" ]' --default_valuation 0.05 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/CSMA/csma2_4_param.nm' --formula 'R{"time"}max=? [ F "all_delivered" ]' --default_valuation 0.05 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_pmc.py --model 'models/pmdp/CSMA/csma2_4_param.nm' --formula 'R{"time"}max=? [ F "all_delivered" ]' --default_valuation 0.05 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/CSMA/csma2_4_param.nm' --formula 'R{"time"}max=? [ F "all_delivered" ]' --default_valuation 0.05 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/CSMA/csma2_4_param.nm' --formula 'R{"time"}max=? [ F "all_delivered" ]' --default_valuation 0.05 --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;

# BENCHMARKS FOR coin
timeout 3600s python3 run_pmc.py --model 'models/pmdp/coin/coin4.pm' --formula 'Pmin=? [ F "all_coins_equal_1" ]' --default_valuation 0.4 --goal_label "{'all_coins_equal_1'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/coin/coin4.pm' --formula 'Pmin=? [ F "all_coins_equal_1" ]' --default_valuation 0.4 --goal_label "{'all_coins_equal_1'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/coin/coin4.pm' --formula 'Pmin=? [ F "all_coins_equal_1" ]' --default_valuation 0.4 --goal_label "{'all_coins_equal_1'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_pmc.py --model 'models/pmdp/coin/coin4.pm' --formula 'Pmin=? [ F "all_coins_equal_1" ]' --default_valuation 0.4 --goal_label "{'all_coins_equal_1'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/coin/coin4.pm' --formula 'Pmin=? [ F "all_coins_equal_1" ]' --default_valuation 0.4 --goal_label "{'all_coins_equal_1'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pmdp/coin/coin4.pm' --formula 'Pmin=? [ F "all_coins_equal_1" ]' --default_valuation 0.4 --goal_label "{'all_coins_equal_1'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;

# BENCHMARKS FOR maze
timeout 3600s python3 run_pmc.py --model 'models/pomdp/maze/maze_simple_extended_m5.drn' --formula 'Rmin=? [F "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pomdp/maze/maze_simple_extended_m5.drn' --formula 'Rmin=? [F "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pomdp/maze/maze_simple_extended_m5.drn' --formula 'Rmin=? [F "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_pmc.py --model 'models/pomdp/maze/maze_simple_extended_m5.drn' --formula 'Rmin=? [F "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pomdp/maze/maze_simple_extended_m5.drn' --formula 'Rmin=? [F "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pomdp/maze/maze_simple_extended_m5.drn' --formula 'Rmin=? [F "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;

# BENCHMARKS FOR drone
timeout 3600s python3 run_pmc.py --model 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn' --formula 'P=? ["notbad" U "goal"]' --goal_label "{'goal','notbad'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn' --formula 'P=? ["notbad" U "goal"]' --goal_label "{'goal','notbad'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn' --formula 'P=? ["notbad" U "goal"]' --goal_label "{'goal','notbad'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_pmc.py --model 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn' --formula 'P=? ["notbad" U "goal"]' --goal_label "{'goal','notbad'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn' --formula 'P=? ["notbad" U "goal"]' --goal_label "{'goal','notbad'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn' --formula 'P=? ["notbad" U "goal"]' --goal_label "{'goal','notbad'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;

# BENCHMARKS FOR satellite
timeout 3600s python3 run_pmc.py --model 'models/satellite/pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06_sparse_full.drn' --formula 'P=? [F "goal"]' --default_valuation 0.01 --goal_label "{'goal'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/satellite/pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06_sparse_full.drn' --formula 'P=? [F "goal"]' --default_valuation 0.01 --goal_label "{'goal'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/satellite/pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06_sparse_full.drn' --formula 'P=? [F "goal"]' --default_valuation 0.01 --goal_label "{'goal'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 1 --explicit_baseline;
timeout 3600s python3 run_pmc.py --model 'models/satellite/pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06_sparse_full.drn' --formula 'P=? [F "goal"]' --default_valuation 0.01 --goal_label "{'goal'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/satellite/pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06_sparse_full.drn' --formula 'P=? [F "goal"]' --default_valuation 0.01 --goal_label "{'goal'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;
timeout 3600s python3 run_prmc.py --model 'models/satellite/pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06_sparse_full.drn' --formula 'P=? [F "goal"]' --default_valuation 0.01 --goal_label "{'goal'}" --validate_delta 1e-3 --output_folder 'output/benchmarks_cav23_partial' --num_deriv 10 --explicit_baseline;

 python3 create_table.py --folder 'output/benchmarks_cav23_partial' --table_name 'tables/benchmarks_cav23_partial' --mode gridworld