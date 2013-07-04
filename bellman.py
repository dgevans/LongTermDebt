# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:41:52 2013

@author: dgevans
"""
import numpy as np
from scipy.optimize import fmin_slsqp
import numdifftools as nd
from scipy.sparse import coo_matrix


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
        qbounds = np.hstack((0.95*self.Para.q_bounds[0],1.05*self.Para.q_bounds[1]))
        bounds = cbounds+[self.Para.xprime_bounds]*S+[self.Para.b_bounds]*S + [qbounds]*S
        [z,minusV,_,imode,smode] = fmin_slsqp(self.objectiveFunction,z0,f_eqcons=self.constraints,bounds=bounds,
                         fprime=self.objectiveFunctionJac,fprime_eqcons=self.constraintsJac,args=(state,),
                         disp=0,full_output=True,iter=10000)
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
        V_q = np.zeros(S)
        for s in range(0,S):
            chi = np.array([ xprime[s],bprime[s],qprime[s] ])
            V_x[s] = self.Vf[s](chi,[1,0,0])
            V_b[s] = self.Vf[s](chi,[0,1,0])
            V_q[s] = self.Vf[s](chi,[0,0,1])
        
        gradobj = Uc(c)*gradc+Ul(l)*gradl + beta*(V_x*gradxprime + V_b*gradbprime + V_q*gradqprime)
        
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
        #first is implimmentability
        impCon = x*uc/(beta*Euc)+b*uc - I(c,l) - xprime - qprime*(bprime-b)
        #promise keeping with phi    
        qCon = q - beta*P[s_,:].dot(qprime + uc)
        return np.hstack((impCon,qCon))
        
        
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
        
        gradI = (c*ucc + uc)*gradc + (l*ull + ul)*gradl
        #first compute the jacobian of the implimentability
        ImpJac = (x*ucc*gradc/(beta*Euc)  - x*uc*gradEuc/(beta*Euc**2) +b*ucc*gradc- gradI - gradxprime -gradqprime*(bprime-b)  - qprime*gradbprime).T
        qconJac = -beta*(gradqprime+ucc*gradc).dot(P)

        return np.vstack((ImpJac,qconJac))
            


class IPOPT_wrapper(object):
    
    def __init__(self,T,state):
        self.T = T
        self.state = state
    
    def eval_f(self,x, user_data = None):
        return self.T.objectiveFunction(x,self.state)
        
    def eval_grad_f(self,x, user_data = None):
        return self.T.objectiveFunctionJac(x,self.state)
    
    def eval_g(self,x, user_data= None):
        return self.T.constraints(x,self.state)
        
    def eval_jac_g(self,x, flag, user_data = None):
        S = len(self.T.Para.P)
        n = S+1
        m = 4*S
        if flag:
            Rows = np.kron(np.arange(0,n),np.ones(m,dtype=np.int))
            Cols = np.kron(np.ones(n,dtype=np.int),np.arange(0,m))
            return Rows,Cols
        else:
            return self.T.constraintsJac(x,self.state).flatten()
            
class IPOPT_wrapper2(object):
    
    def __init__(self,T,state,z0):
        self.T = T
        self.state = state
        self.z0 = z0
    
    def objective(self,x):
        return self.T.objectiveFunction(x,self.state)
        
    def gradient(self,x):
        return self.T.objectiveFunctionJac(x,self.state)
    
    def constraints(self,x):
        return self.T.constraints(x,self.state)
        
    def jacobian(self,x):
         res =  self.T.constraintsJac(x,self.state).flatten()
         #pdb.set_trace()
         return res
    
    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        # 
        S = len(self.T.Para.P)
        hs = coo_matrix(np.tril(np.ones((4*S, 4*S))))
        return (hs.col, hs.row)
        
        
    def hessian(self, x, lagrange, obj_factor):
        '''
        Computing the hessian with finite differences for test
        '''
        H = obj_factor*nd.Jacobian(self.gradient)(x)
        S = len(self.T.Para.P)
        for i in range(0,S+1):
            f_i = lambda z: self.T.constraintsJac(x,self.state)[i,:]
            Jf_i = nd.Jacobian(f_i)
            H += lagrange[i]*Jf_i(x)
        hs = coo_matrix(np.tril(np.ones((4*S, 4*S))))
        return H[hs.row, hs.col]
        
    def intermediate(
            self, 
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print "Objective value at iteration #%d is - %g" % (iter_count, obj_value)
        print inf_pr
        print inf_du