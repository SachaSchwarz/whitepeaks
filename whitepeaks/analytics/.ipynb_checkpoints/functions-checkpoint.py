'''
###############################################################################
FUNCTIONS MODULE
###############################################################################

This module contains the following functions:

Polynomial functions
--------------------
> poly1(x,A1,x0,k)
> poly2(x,A2,A1,x0,k)
> poly3(x,A3,A2,A1,x0,k)
> poly2D(x,k,C1,B1,A1,C2,B2,A2,x0,y0)
> wpoly2d(x,A1,A2,B1,B2,C1,C2,k2)

Gaussian functions
------------------
> gauss(x,a,x0,sigma,k)
> gauss2d(xy,a,x0,y0,sigma_x,sigma_y,rho,k)

Polynomial and Gaussian cost functions
--------------------------------------
> poly2D_cost(p,x,y,z)
> wpoly2d_cost(p,x,y,A,z)
> gauss_cost(p,x,y)
> gauss2d_cost(p,x,y,z)
> gauss2d_residual(p,x,y,z)

'''

import numpy as np

'''
-------------------------------------------------------------------------------
Polynomial functions
-------------------------------------------------------------------------------
'''
def poly1(x,A1,x0,k):
    '''
    1nd order polynomial
    '''
    return A1*(x-x0)+k

def poly2(x,A2,A1,x0,k):
    '''
    2nd order polynomial
    '''
    return A2*(x-x0)**2+A1*(x-x0)+k

def poly3(x,A3,A2,A1,x0,k):
    '''
    3rd order polynomial
    '''
    return A3*(x-x0)**3+A2*(x-x0)**2+A1*(x-x0)+k

def poly2D(x,k,C1,B1,A1,C2,B2,A2,x0,y0):
    X,Y=x
    return (A1*(X-x0)**3+A2*(Y-y0)**3
           +B1*(X-x0)**2+B2*(Y-y0)**2 
           +C1*(X-x0) + C2*(Y-y0) +k)

def wpoly2d(x,A1,A2,B1,B2,C1,C2,k2):
    '''
    Weighted polynomial.
    '''

    X,Y,weights=x

    return weights*(C1*X**3+C2*Y**3
        +B1*X**2+B2*Y**2 
        +A1*X + A2*Y +k2)

'''
-------------------------------------------------------------------------------
Gaussian functions
-------------------------------------------------------------------------------
'''
def gauss(x,a,x0,sigma,k):
    '''
    1D Gaussian function

    Parameters
    ----------
    x: ndarray
        1D array
    a: float
        Amplitude.
    x0: float 
        Centre.
    sigma: float
        Standard deviation (RMS).
    k: float
        Vertical Offset.

    Returns
    ----------
    out:ndarray
        1D array of y values for input x
    '''
    return a*np.exp(-(x-x0)**2.0/(2.0*sigma**2.0))+k

def gauss2d(xy,a,x0,y0,sigma_x,sigma_y,rho,k):
    '''
    2D Correlated Gaussian function

    Parameters
    ----------
    xy: tuple(ndarray,ndarray)
        (X,Y) meshgrid
    a: float
        Amplitude.
    x0: float 
        Centre along x axis.
    y0: float 
        Centre along y axis.
    sigma_x: float
        Standard deviation (RMS).
    sigma_y: float
        Standard deviation (RMS).
    rho: float
        Statistical correlations, between -1 and 1.
    k: float
        Vertical offset.

    Returns
    ----------
    out:ndarray
        1D array of y values for input x
    '''

    X,Y=xy
    return a*(np.exp(
        -1.0/(1.0-rho**2.0)*((X-x0)**2.0/(2.0*sigma_x**2.0)
                        +(Y-y0)**2.0/(2.0*sigma_y**2.0)
                        -rho*(X-x0)*(Y-y0)/(sigma_x*sigma_y)  
                        )))+k

'''
-------------------------------------------------------------------------------
Polynomial and Gaussian cost functions
-------------------------------------------------------------------------------
'''
def poly2D_cost(p,x,y,z):
    '''
    Cost function for poly fit. To be used with
    scipy.optimize.least_squares fitting function.
    '''
    return (z-poly2D([x,y],*p))#/np.sqrt(poly2D([x,y],*p)+0.01))

def wpoly2d_cost(p,x,y,A,z):
    return z-wpoly2d([x,y,A],*p)

def gauss_cost(p,x,y):
    '''
    Calculate cost of function. For use with scipy.optimize.least_squares

    Parameters
    ----------
    p: list 
        Parameters of the function.
    x: ndarray 
        x values
    y: ndarray 
        yvalues

    Returns
    ----------
    out: ndarray

    '''
    return ((y-gauss(x,*p))/np.sqrt(gauss(x,*p)+0.01))

def gauss2d_cost(p,x,y,z):
    '''
    Cost function for gaussian fit. To be used with
    scipy.optimize.least_squares fitting function.
    '''
    return ((z-gauss2d([x,y],*p))/np.sqrt(gauss2d([x,y],*p)+0.01))

def gauss2d_residual(p,x,y,z):
    '''
    Residual for gaussian fit.
    '''
    return np.sum((z-gauss2d([x,y],*p))**2/(gauss2d([x,y],*p)+0.01))