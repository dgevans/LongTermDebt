# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:35:47 2013

@author: dgevans
"""

__author__ = 'dgevans'
import numpy as np
from scipy.optimize import root

def LSResiduals(z,mu,Para):
    S = Para.P.shape[0]
    c = z[0:S]
    l = z[S:2*S]
    uc = Para.Uc(c)
    ucc = Para.Ucc(c)
    ul = Para.Ul(l)
    ull = Para.Ull(l)

    res = Para.theta*l-c-Para.g
    foc_c = uc -(uc+ucc*c)*mu
    foc_l = (ul-(ul+ull*l)*mu)/Para.theta

    return np.hstack((res,foc_c+foc_l))

def solveLSmu(mu,Para):
    S = Para.P.shape[0]
    f = lambda z: LSResiduals(z,mu,Para)

    z_mu = root(f,0.5*np.ones(2*S)).x

    return z_mu[0:S],z_mu[S:2*S]

def solveLucasStockey(x,s_,Para):
    S = Para.P.shape[0]
    def x_mu(mu):
        c,l = solveLSmu(mu,Para)
        I = Para.I(c,l)
        return Para.beta*Para.P[s_,:].dot(np.linalg.solve(np.eye(S)-(Para.beta*Para.P.T).T,I))

    mu_SL = root(lambda mu: x_mu(mu)-x,0).x

    return solveLSmu(mu_SL,Para)

    
def LSxmu(x,mu,Para):
    c,l = solveLSmu(mu,Para)
    I = Para.I(c,l)
    uc = Para.U.uc(c,l,Para)
    Euc= Para.P[0,:].dot(uc)
    return x*uc/(Para.beta*Euc)-I