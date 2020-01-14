'''
###############################################################################
STATS MODULE
###############################################################################

This module contains the following functions:

> get_moments(x,y)
> get_marginal(X,Z)
> correlation(x,y,z)

'''

import numpy as np 

def get_moments(x,y):
    '''
    Calculate statistical moments of a 1D distribution. 
    
    Parameters 
    ----------
    x: ndarray
        x coordinates of the distribution.
    y: ndarray
        y coordinates of the distribution.

    Returns
    ----------
    out:tuple (float, float)
        Mean and standard deviation of distribution.
    '''

    xmean=np.sum(y*x)/np.sum(y)
    xsigma=np.sqrt(1.0/np.sum(y)*np.sum(y*(x-xmean)**2))
    return xmean,xsigma

def get_marginal(X,Z,
                 axis=0):
    '''
    Calculate the 1D marginal along the one axis of a 2D distribution. 
    
    Parameters 
    ----------
    X: ndarray
        2D meshgrid array or X coordinates.
    Z: ndarray
        2D array of amplitudes y coordinates of the distribution.
    axis:{0,1},optional
        Axis over which to take the marginal.

    Returns
    ----------
    out:tuple (ndarray, ndarray)
        1D array of coordinates and marginal values at those coordinates.
    '''

    xsum=np.sum(Z,axis=axis)
    dx=np.diff(X,axis=axis^1)[0][0]
    #xlin=np.arange(X.min(),X.max()+dx,dx)
    xlin=np.linspace(X.min(),X.max(),xsum.size)
    return xlin,xsum

def correlation(x,y,z):
    '''
    Calculate the statistical correlation a 2D distribution. 
    
    Parameters 
    ----------
    x: ndarray
        1D array of x coordinates.
    y: ndarray
        1D array of y coordinates.
    z: ndarray
        1D array of z coordinates.

    Returns
    ----------
    out:float
        Statistical correlation of the distribution.
    '''

    x0,xsigma=get_moments(x,z)
    y0,ysigma=get_moments(y,z)

    rho=1/np.sum(z)*np.sum(z*(x-x0)*(y-y0))/(xsigma*ysigma)

    return rho
