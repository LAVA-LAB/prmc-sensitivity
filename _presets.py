#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:55:11 2023

@author: thom
"""

args.model = 'models/slipgrid/double_pmc_size=400_params=1000_seed=0.drn'
args.parameters = 'models/slipgrid/double_pmc_size=400_params=1000_seed=0.json'
args.formula = 'Rmin=? [F "goal"]'

# args.model = 'models/pdtmc/brp64_4.pm'
# args.formula = 'P=? [ F s=5 ]'
# args.default_valuation = 0.9
# args.goal_label = {'(s = 5)'}

# args.model = 'models/pmdp/brp/brp.pm'
# args.formula = 'Pmax=? [ F (s=5) ]'
# args.default_valuation = 0.9
# args.goal_label = {'(s = 5)'}

# args.model = 'models/pmdp/coin/coin4.pm'
# args.formula = 'Pmin=? [ F "finished" & "all_coins_equal_1" ]'
# args.goal_label = {'finished', 'all_coins_equal_1'}

# args.model = 'models/pmdp/CSMA/csma2_4_param.nm'
# args.formula = 'R{"time"}max=? [ F "all_delivered" ]'
# args.default_valuation = 0.1
# args.goal_label = {'all_delivered'}

# args.model = 'models/pomdp/drone/pomdp_drone_4-2-mem5-simple.drn'
# args.formula = 'P=? ["notbad" U "goal"]'
# args.goal_label = {'goal'}

args.pMC_engine = 'spsolve'
args.num_deriv = 1

args.validate_delta  = 1e-5