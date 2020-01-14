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

#Polarization states
H=np.matrix([[1.],[0.]]) 
V=np.matrix([[0.],[1.]]) 
D=1/np.sqrt(2)*np.matrix([[1],[1]]) 
A=1/np.sqrt(2)*np.matrix([[1],[-1]]) 
L=1/np.sqrt(2)*np.matrix([[1],[1j]]) 
R=1/np.sqrt(2)*np.matrix([[1],[-1j]]) 

#Pauli matrices
X=np.matrix([[0,1],[1,0]]) #{{{
Y=np.matrix([[0,-1j],[1j,0]]) 
Z=np.matrix([[1,0],[0,-1]]) #}}}

def DM(state):
    '''
    Calculate the density matrix from a pure state.  

    Parameters
    ----------
    state:ndarray 

    Returns
    ----------
    out:matrix
        
    '''
    return np.kron(state,state.H) 

#Jones Matrices
def Rt(theta):#{{{
    '''
    Rotation matrix. 

    Parameters
    ----------
    theta: float 
      Angle in radians.

    Returns
    ----------
    out:matrix
        
    '''
    return np.matrix([[np.cos(theta),-np.sin(theta)
                      ],[np.sin(theta),np.cos(theta)]]) #}}}

def HWP(theta):#{{{
    '''
    Rotation matrix for half-wave plate at angle theta.

    Parameters
    ----------
    theta: float 
       Wave plate angle in radians.

    Returns
    ----------
    out:matrix
        
    '''
    return (Rt(theta))*(np.matrix([[1j,0],[0,-1j]])*Rt(theta).H) #}}}

def QWP(theta):#{{{
    '''
    Rotation matrix for quarter-wave plate at angle theta.

    Parameters
    ----------
    theta: float 
       Wave plate angle in radians.

    Returns
    ----------
    out:matrix
        
    '''
    return (Rt(theta)
            *np.matrix([[np.exp(1j*np.pi/4) ,0],
                        [0,np.exp(-1j*np.pi/4)]])*(Rt(theta).H)) #}}}
