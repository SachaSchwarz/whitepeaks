'''
###############################################################################
WAVEPLATES MODULE
###############################################################################

This module contains the following functions:

> DM(state)
> Rt(theta)
> HWP(theta)
> QWP(theta)

'''

import numpy as np

#States
H=np.matrix([[1.],[0.]]) 
V=np.matrix([[0.],[1.]]) 
D=1/np.sqrt(2)*np.matrix([[1],[1]]) 
A=1/np.sqrt(2)*np.matrix([[1],[-1]]) 
L=1/np.sqrt(2)*np.matrix([[1],[1j]]) 
R=1/np.sqrt(2)*np.matrix([[1],[-1j]]) 

#Paulis
X=np.matrix([[0,1],[1,0]]) #{{{
Y=np.matrix([[0,-1j],[1j,0]]) 
Z=np.matrix([[1,0],[0,-1]]) #}}}

def DM(state):
    return np.kron(state,state.H) 

#Jones Matrices
def Rt(theta):#{{{
    return np.matrix([[np.cos(theta),-np.sin(theta)
                      ],[np.sin(theta),np.cos(theta)]]) #}}}

def HWP(theta):#{{{
    '''
    Half wave plate at angle theta
    '''
    return (Rt(theta))*(np.matrix([[1j,0],[0,-1j]])*Rt(theta).H) #}}}

def QWP(theta):#{{{
    '''
    Quarter Wave Plate at angle theta
    '''
    return (Rt(theta)
            *np.matrix([[np.exp(1j*np.pi/4) ,0],
                        [0,np.exp(-1j*np.pi/4)]])*(Rt(theta).H)) #}}}
