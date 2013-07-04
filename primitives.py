# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:27:56 2013

@author: dgevans
"""
import numpy as np


class parameters(object):
    '''
    Holds the basic parameters
    '''
    #productivity    
    theta = np.array([1.0])
    
    #government expenditure
    g = np.array([0.15,0.17])
    
    #Markov process transition matrix
    P = 0.5*np.ones((2,2))
    
    #discount factor
    beta = np.array([0.95])
    

class CES_parameters(parameters):
    
    sigma = 2.0
    
    gamma = 2.0
    
    def U(self,c,l):
        if self.sigma == 1:
            return np.log(c)-l**(1.0+self.gamma)/(1.0+self.gamma)
        else:
            return c**(1.0-self.sigma)/(1.0 -self.sigma)-l**(1.0+self.gamma)/(1.0+self.gamma)
            
    def Uc(self,c):
        return c**(-self.sigma)
        
    def Ul(self,l):
        return -l**(self.gamma)
        
    def Ucc(self,c):
        return -self.sigma*c**(-self.sigma-1.0)
        
    def Ull(self,l):
        return -self.gamma*l**(self.gamma-1)
        
    def I(self,c,l):
        return self.Uc(c)*c+self.Ul(l)*l
        
        
class BGP_parameters(parameters):
    
    sigma_1 = 1.0
    
    sigma_2 = 1.0
    
    psi = 0.6958
    
    def U(self,c,l):
        if self.sigma_1 == 1:
            U = self.psi*np.log(c)
        else:
            U = self.psi*c**(1.0-self.sigma_1)/(1.0-self.sigma_1)
        if self.sigma_2 == 1:
            U += (1.0-self.psi)*np.log(1.0-l)
        else:
            U += (1.0-self.psi)*(1.0-l)**(1.0-self.sigma_2)/(1.0-self.sigma_2)
        return U
        
    def Uc(self,c):
        return self.psi*c**(-self.sigma_1)
    
    def Ul(self,l):
        return -(1.0-self.psi)*(1.0-l)**(-self.sigma_2)
    
    def Ucc(self,c):
        return -self.psi*self.sigma_1*c**(-self.sigma_1-1.0)
        
    def Ull(self,l):
        return -(1.0-self.psi)*self.sigma_2*(1.0-l)**(-self.sigma_2-1.0)
        
    def I(self,c,l):
        return self.Uc(c)*c+self.Ul(l)*l