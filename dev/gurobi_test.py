# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:03:30 2023

@author: Thom Badings
"""

import gurobipy as gp

m = gp.Model('CVX')

x = m.addVar()
y = m.addVar(lb=5, ub=5)
cns = m.addConstr(x * (y * y * y) <= 10)

m.setObjective(y, gp.GRB.MAXIMIZE)

m.optimize()

y.lb = y.ub = 10

m.optimize()