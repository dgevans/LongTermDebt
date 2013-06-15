# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 08:47:22 2013

@author: dgevans
"""
import complete as CM
import numpy as np
import utilities

def CMValueFunction(state,Para):
    '''
    Computes the complete markets value function as a funcion of state
    '''
    (x,b,q),s_ = state
    
    c,l = CM.solveLucasStockey(x+q*b,s_,Para)
    
    S = len(Para.P)
    
    return Para.P[s_,:].dot( np.linalg.solve(np.eye(S)-(Para.beta*Para.P).T,Para.U(c,l)) ),c
    
    
def InitializeValueFunction(Para):
    '''
    Initializes the value function 
    '''
    
    def V0(state):
        
        S = len(Para.P)
        V,c = CMValueFunction(state,Para)
        (x,b,q),s_ = state
        xprime = x*np.ones(S)
        bprime = b*np.ones(S)
        qprime = q*np.ones(S)
        return np.hstack( (c,xprime,bprime,qprime,V) )
        
    policies = np.vstack(map(V0,Para.domain))
    
    return utilities.fitValueFunctionAndPolicies(Para.domain,policies,Para)