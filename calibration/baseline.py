# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:15:18 2013

@author: dgevans
"""

import primitives
import numpy as np
from Spline import Spline

Para = primitives.BGP_parameters()

Para.theta = np.array([3.3])

Para.g = np.array([.35,.37])

Para.xprime_bounds = np.array([-1.,1.])

Para.q_bounds = np.array([Para.Uc(2.1),Para.Uc(1.9)])*Para.beta/(1-Para.beta)

Para.b_bounds = Para.xprime_bounds/(Para.q_bounds[1])

xGrid = np.linspace(Para.xprime_bounds[0],Para.xprime_bounds[1],10)
bGrid = np.linspace(Para.b_bounds[0],Para.b_bounds[1],10)
qGrid = np.linspace(Para.q_bounds[0],Para.q_bounds[1],10)

X = Spline.makeGrid((xGrid,bGrid,qGrid))

S = len(Para.P)
slist = []
for s in range(0,S):
    slist += [s]*len(X)

Para.domain = zip(np.vstack([X]*S),slist)

Para.deg = [2,2,2]