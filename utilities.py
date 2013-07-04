# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 09:02:45 2013

@author: dgevans
"""
import numpy as np
from Spline import Spline


#Holds value function truly it actually fits the certain consumption equivalent as that will be closer to linear
class ValueFunctionSpline:
    def __init__(self,X,y,k,sigma,beta):
        self.sigma = sigma
        self.beta = beta
        if sigma == 1.0:
            y = np.exp((1-beta)*y)
        else:
            y = ((1-beta)*(1.0-sigma)*y)**(1.0/(1.0-sigma))
        self.f = Spline(X,y,k)

    def fit(self,X,y,k):
        if self.sigma == 1.0:
            y = np.exp((1-self.beta)*y)
        else:
            y = ((1-self.beta)*(1.0-self.sigma)*y)**(1.0/(1.0-self.sigma))
        self.f.fit(X,y,k)

    def getCoeffs(self):
        return self.f.getCoeffs()

    def __call__(self,X,d = None):
        if d==None:
            if self.sigma == 1.0:
                return np.log(self.f(X,d))/(1-self.beta)
            else:
                return (self.f(X,d)**(1.0-self.sigma))/((1.0-self.sigma)*(1-self.beta))

        if sum(d)==1:
            return self.f(X)**(-self.sigma) * self.f(X,d)/(1-self.beta)
        raise Exception('Error: d must equal None or 1')
        
        
class ValueFunction(object):
    
    def __init__(self,Vf,c_policy,xprime_u_policy,xprime_s_policy,qprime_policy):
        self.Vf = Vf
        self.c_policy = c_policy
        self.xprime_policy = xprime_u_policy
        self.bprime_policy = xprime_s_policy
        self.qprime_policy = qprime_policy
        
    def __getitem__(self,i):
        return self.Vf[i]

def fitValueFunctionAndPolicies(domain,policies,Para):
    '''
    Fits the Value Function and Polciies given domain and list of policies
    '''
    S = len(Para.P.shape)
    merged = zip(domain,policies) #merge policies with domain so we can pull out policie that did not work
    c_policy,xprime_policy,bprime_policy,qprime_policy = {},{},{},{} #initialize new policies as empty dicts
    Vf = []
    
    for s_ in range(0,S):
        #filter out the states where the maximization does not work
        s_domain,s_policies = zip(* filter(lambda x: x[1] != None and x[0][1]==s_, merged) )
        X,_ = zip(*s_domain)
        X = np.vstack(X)
        s_policies = np.vstack(s_policies)
        cNew,xprimeNew,bprimeNew,qprimeNew,Vnew =s_policies[:,0:S],s_policies[:,S:2*S],s_policies[:,2*S:3*S],s_policies[:,3*S:4*S],s_policies[:,4*S]#unpack s_policis and then stacks them into a matrix     
        
        #fit policies
        Vf.append(ValueFunctionSpline(X,Vnew,Para.deg,Para.sigma_1,Para.beta))
        for s in range(0,S):
            c_policy[(s_,s)] = Spline(X,cNew[:,s],[1,1,1])
            xprime_policy[(s_,s)] = Spline(X,xprimeNew[:,s],[1,1,1])
            bprime_policy[(s_,s)] = Spline(X,bprimeNew[:,s],[1,1,1])
            qprime_policy[(s_,s)] = Spline(X,qprimeNew[:,s],[1,1,1])
            
            
    return ValueFunction(Vf,c_policy,xprime_policy,bprime_policy,qprime_policy)
    
def Df(f,z0):
    '''
    Computes the Hessian of an object
    '''
    f0 = f(z0)
    m = f0.size
    n = z0.size
    Df = np.zeros((n,m))
    h = 1e-6
    I = np.eye(n)
    for i in range(0,n):
        dz = I[i,:]*h
        Df[i,:] = (f(z0+dz)-f0)/h
    return Df