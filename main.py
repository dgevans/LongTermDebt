# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:43:02 2013

@author: dgevans
"""

from calibration import baseline
import bellman
import utilities
import initialize
import numpy as np
import sys

Para = baseline.Para

Vf,policiesold = initialize.InitializeValueFunction(Para)

T = bellman.BellmanMap(Para)

S = len(Para.P)
cold = []
for s in range(0,S):
    cold.append(Vf[s].getCoeffs())
    
for t in range(0,1000):
    Vnew = T(Vf)
    policies = np.vstack(map(Vnew,Para.domain))
    policies = 0.9*policies+0.1*policiesold
    Vf = utilities.fitValueFunctionAndPolicies(Para.domain,policies,Para)
    c = []
    diff = 0
    for s in range(0,S):
        c.append(Vf[s].getCoeffs())
        diff = max(diff,np.max(np.abs(c[s]-cold[s])))
    print np.argmax(np.abs(c[0]-cold[0]))
    print diff
    sys.stdout.flush()
    cold = c
    policiesold = policies
    