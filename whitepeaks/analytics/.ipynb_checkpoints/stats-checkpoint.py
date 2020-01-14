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
    xmean=np.sum(y*x)/np.sum(y)
    xsigma=np.sqrt(1.0/np.sum(y)*np.sum(y*(x-xmean)**2))
    return xmean,xsigma

def get_marginal(X,Z,
                 axis=0):

    xsum=np.sum(Z,axis=axis)
    dx=np.diff(X,axis=axis^1)[0][0]
    #xlin=np.arange(X.min(),X.max()+dx,dx)
    xlin=np.linspace(X.min(),X.max(),xsum.size)
    return xlin,xsum

def correlation(x,y,z):

    x0,xsigma=get_moments(x,z)
    y0,ysigma=get_moments(y,z)

    rho=1/np.sum(z)*np.sum(z*(x-x0)*(y-y0))/(xsigma*ysigma)

    return rho