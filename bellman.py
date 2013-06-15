# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:41:52 2013

@author: dgevans
"""
import numpy as np
from scipy.optimize import fmin_slsqp

class BellmanMap(object):
    '''
    Class holding the bellman map
    '''
    
    def __init__(self,Para):
        self.Para = Para
        
    def __call__(self,Vf):
        self.Vf = Vf
        self.c_policy = Vf.c_policy
        self.xprime_policy = Vf.xprime_policy
        self.bprime_policy = Vf.bprime_policy
        self.qprime_policy = Vf.qprime_policy
        return self.maximizeObjective
    
    def maximizeObjective(self,state,z0=None):
        '''
        Maximizes the objective returning objective values and policies
        '''
        chi,s_ = state
        S = len(self.Para.P)
        
        #fill initial guess from previous periods policies
        if z0==None:
            z0 = np.zeros(4*S)
            for s in range(0,S):
                z0[s] = self.c_policy[s_,s](chi)
                z0[S+s] = self.xprime_policy[s_,s](chi)
                z0[2*S+s] = self.bprime_policy[s_,s](chi)
                z0[3*S+s] = self.qprime_policy[s_,s](chi)
        
        #now perform optimization
        cbounds = []
        cub = self.Para.theta-self.Para.g
        for s in range(0,S):
            cbounds.append([0,cub[s]])
        bounds = cbounds+[self.Para.xprime_bounds]*S+[self.Para.b_bounds]*S + [self.Para.q_bounds]*S
        [z,minusV,_,imode,smode] = fmin_slsqp(self.objectiveFunction,z0,f_eqcons=self.constraints,bounds=bounds,
                         fprime=self.objectiveFunctionJac,fprime_eqcons=self.constraintsJac,args=(state,),
                         disp=0,full_output=True,iter=1000)
        if imode !=0:
            print smode
            return None
        return np.hstack((z,-minusV))
        
        
    
    def objectiveFunction(self,z,state):
        '''
        Objective function to minize
        '''
        (x,b,_),s_ = state
        P = self.Para.P
        S = len(P)
        g = self.Para.g
        beta = self.Para.beta
        theta = self.Para.theta
        
        c = z[0:S]
        l = ( g + c )/theta
        xprime = z[S:2*S]
        bprime = z[2*S:3*S]
        qprime = z[3*S:4*S]
        
        Vprime = np.zeros(S)
        for s in range(0,S):
            Vprime[s] = self.Vf[s]([xprime[s],bprime[s],qprime[s]])
        
        obj = self.Para.U(c,l) + beta*Vprime
        
        return -P[s_,:].dot(obj)
        
    def objectiveFunctionJac(self,z,state):
        '''
        Gradient of the objective function
        '''
        (x,b,_),s_ = state
        P = self.Para.P
        S = len(P)
        g = self.Para.g
        Uc = self.Para.Uc
        Ul = self.Para.Ul        
        
        beta = self.Para.beta
        theta = self.Para.theta
        gradMat = np.kron(np.eye(4),np.eye(S))        
        
        c = z[0:S]
        gradc = gradMat[:,0:S]
        l = ( g + c )/theta
        gradl = gradc/theta
        
        xprime = z[S:2*S]
        gradxprime = gradMat[:,S:2*S]
        bprime = z[2*S:3*S]
        gradbprime = gradMat[:,2*S:3*S]
        qprime = z[3*S:4*S]
        gradqprime = gradMat[:,3*S:4*S]
                   
        V_x = np.zeros(S)
        V_b = np.zeros(S)
        V_phi = np.zeros(S)
        for s in range(0,S):
            chi = np.array([ xprime[s],bprime[s],qprime[s] ])
            V_x[s] = self.Vf[s](chi,[1,0,0])
            V_b[s] = self.Vf[s](chi,[0,1,0])
            V_phi[s] = self.Vf[s](chi,[0,0,1])
        
        gradobj = Uc(c)*gradc+Ul(l)*gradl + beta*(V_x*gradxprime + V_b*gradbprime + V_b*gradqprime)
        
        return -P[s_,:].dot(gradobj.T)
        
        
    def constraints(self,z,state):
        '''
        Contraints of the bellman equation
        '''
        (x,b,q),s_ = state
        P = self.Para.P
        S = len(P)
        g = self.Para.g
        theta = self.Para.theta
        I = self.Para.I
        beta= self.Para.beta
        
        
        c = z[0:S]
        uc = self.Para.Uc(c)
        Euc = self.Para.P[s_,:].dot(uc)
        l = ( g + c )/theta
        xprime = z[S:2*S]
        bprime = z[2*S:3*S]
        qprime = z[3*S:4*S]
        
        cons = np.zeros(S+1)
        #first is implimmentability
        cons[0:S] = x*uc/(beta*Euc)+b*uc - I(c,l) - xprime - qprime*(bprime-b)
        #promise keeping with phi
        cons[S] = q - beta*P[s_,:].dot(qprime + uc)
        
        return cons
        
        
    def constraintsJac(self,z,state):
        '''
        Computes the jacobain of the constraints
        '''
        (x,b,q),s_ = state
        P = self.Para.P[s_,:]
        beta = self.Para.beta
        S = len(P)
        g = self.Para.g
        Uc = self.Para.Uc
        Ucc = self.Para.Ucc
        Ul = self.Para.Ul    
        Ull = self.Para.Ull
        
        theta = self.Para.theta
        
        gradMat = np.kron(np.eye(4),np.eye(S))        
        
        c = z[0:S]
        gradc = gradMat[:,0:S]
        uc = Uc(c)
        ucc = Ucc(c)
        Euc = P.dot(uc)
        gradEuc = (ucc*gradc).dot(P).reshape(-1,1)        
        
        l = ( g + c )/theta
        ul = Ul(l)
        ull = Ull(l)
        gradl = gradc/theta
        
        gradxprime = gradMat[:,S:2*S]
        bprime = z[2*S:3*S]
        gradbprime = gradMat[:,2*S:3*S]
        qprime = z[3*S:4*S]
        gradqprime = gradMat[:,3*S:4*S]
        
        gradI = (c*ucc + uc)*gradc - (l*ull + ul)*gradl
                   
        consJac = np.zeros((S+1,len(z)))
        #first compute the jacobian of the implimentability
        consJac[0:S,:] = (x*ucc*gradc/(beta*Euc)  - x*uc*gradEuc/(beta*Euc**2) + b*ucc*gradc\
            - gradI - gradxprime - gradqprime*(bprime-b) - qprime*gradbprime).T
        consJac[S,:] = beta*(gradqprime+ucc*gradc).dot(P)

        return consJac        
        