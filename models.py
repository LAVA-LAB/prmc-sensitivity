# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:17:53 2022

@author: Thom Badings
"""

import cvxpy as cp
from poly import pol

def IMC_3state():

    params = {
        'par1': cp.Parameter(value = 0.2),
        'par2': cp.Parameter(value = 0.2),
        'par3': cp.Parameter(value = 0.9),
        #
        'parA': cp.Parameter(value = 0.1 + 1e-2),
        'parB': cp.Parameter(value = 0.5),
        }
    
    states = set({0,1,2,3})
    
    edges = {
        (0,1): [pol(params['par1'], [0, 1]), pol(params['parA'], [0.35, -1])],
        (0,2): [pol(params['par2'], [0, 1]), pol(params['parA'], [0.35, -1])],
        (0,3): [pol(params['parB'], [0, 1]), pol(params['par3'], [0, 1])],
        }
    
    # Initial state index
    reward = [0,2,2,3]
    
    sI = 0
    
    return params, states, edges, reward, sI

def IMC1():

    params = {
        'par1': cp.Parameter(value = 0.4),
        'par2': cp.Parameter(value = 0.6),
        'par3': cp.Parameter(value = 0.4),
        'par4': cp.Parameter(value = 0.6),
        #
        'par5': cp.Parameter(value = 0.2),
        'par6': cp.Parameter(value = 0.5),
        'par7': cp.Parameter(value = 0.3),
        'par8': cp.Parameter(value = 0.6),
        #
        'par9': cp.Parameter(value = 0.35),
        'par10': cp.Parameter(value = 1.01),
        #
        'par11': cp.Parameter(value = 0.35),
        'par12': cp.Parameter(value = 1.01),
        }
    
    states = set({0,1,2,3,4})
    
    edges = {
        (0,1): [pol(params['par1'], [0, 1]), pol(params['par2'], [0, 1])],
        (0,2): [pol(params['par3'], [0, 1]), pol(params['par4'], [0, 1])],
        (1,2): [pol(params['par5'], [0, 1]), pol(params['par6'], [0, 1])],
        (1,3): [pol(params['par7'], [0, 1]), pol(params['par8'], [0, 1])],
        (2,4): [pol(params['par9'], [0, 1]), pol(params['par10'], [0, 1])],
        (3,4): [pol(params['par11'], [0, 1]), pol(params['par12'], [0, 1])],
        }
    
    # Initial state index
    reward = [0,1,2,3,4]
    
    sI = 0
    
    return params, states, edges, reward, sI