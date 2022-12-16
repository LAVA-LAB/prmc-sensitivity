# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:17:53 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np
from poly import poly

class prMDP:
    
    def __init__(self, S, sI, Act, V, P, R):
        
        self.S = S
        self.sI = sI
        self.Act = Act
        self.V = V
        self.P = P
        self.R = R

        self.term = set([s for s in self.S if not 
                         any([(s,a) in self.P for a in self.Act]) ])
        self.Snonterm = self.S - self.term
        


def prMDP_reza():
    
    states = set({0,1,2,3})
    actions = set({0})
    
    params = {
        'alpha': cp.Parameter(value = 0.11),
        'beta': cp.Parameter(value = 0.11),
        }
    
    sp = [1,2,3]
    
    A = np.vstack((
            np.kron(np.eye(len(sp)), np.array([[-1],[1]])),
        ))

    b = np.array([
        -0.1,
        poly(params['alpha'], {0: 0.35, 1: -1}),
        -0.1,
        poly(params['alpha'], {0: 0.35, 1: -1}),
        poly(params['beta'], {0: -0.3, 1: -2}),
        0.9,
        ])
    
    transfunc = {
        (0,0): {'sp': sp, 'A': A, 'b': b}
        }
    
    reward = [0,0,0,1]
    
    sI = {0: 1}
    
    policy = {
        0: {0: 1}
        }
    
    M = prMDP(states, sI, actions, params, transfunc, reward)
    
    return M, policy



def prMDP_3S():
    
    states = set({0,1,2,3})
    actions = set({0})
    
    params = {
        '01_low': cp.Parameter(value = 0.4),
        '01_upp': cp.Parameter(value = 0.6),
        '02_low': cp.Parameter(value = 0.3),
        '02_upp': cp.Parameter(value = 0.55),
        '03_low': cp.Parameter(value = 0.1),
        '03_upp': cp.Parameter(value = 0.5),
        }
    
    sp = [1,2,3]
    
    A = np.vstack((
            np.kron(np.eye(len(sp)), np.array([[-1],[1]])),
        ))

    b = np.array([
        poly(params['01_low'], {1: -1}),
        poly(params['01_upp'], {1: 1}),
        poly(params['02_low'], {1: -1}),
        poly(params['02_upp'], {1: 1}),
        poly(params['03_low'], {1: -1}),
        poly(params['03_upp'], {1: 1}),
        # -1, 1
        ])
    
    transfunc = {
        (0,0): {'sp': sp, 'A': A, 'b': b}
        }
    
    reward = [0,2,2,3]
    
    sI = {0: 1}
    
    policy = {
        0: {0: 1}
        }
    
    M = prMDP(states, sI, actions, params, transfunc, reward)
    
    return M, policy

    
    
    

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