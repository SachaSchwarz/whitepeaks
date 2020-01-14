'''
###############################################################################
MEASURE MODULE
###############################################################################

This module contains the following functions:

> fidelity(F,G,x)
> NRMS(data,estimate)
> chi2_reduced(data,estimate)
> error_I(data,estimate)
> error_phi(data,estimate)
> BhattacharyyaDistance(p,q)
> BhattacharyyaCoeff(p,q)
> random_phase(X)
> FROG_error(data,estimate)

'''

import numpy as np

from scipy.optimize import least_squares 

def fidelity(F,G,x):
    '''
    Calculate the fidelity between two state F and G over axis x.

    Parameters
    ----------
    F: ndarray
        1D or 2D array.

    G: ndarray
        1D or 2D array

    x: ndarray or tuple(ndarray,ndarray)
        Array over which to integrate. If F and G are 2D, specify both
        axes x=(x,y) over which to integrate. 

    Returns
    ----------
    out:int
        Fidelity between F and G. 

    '''
    # If 2D
    if np.size(np.shape(F))==2:
        x,y=x
        normF=np.sqrt(np.trapz(np.trapz(np.abs(F)**2,x=x),x=y))
        normG=np.sqrt(np.trapz(np.trapz(np.abs(G)**2,x=x),x=y))
    
        return 1.0/(normF*normG)**2*np.abs(np.trapz(np.trapz(F*np.conjugate(G),x=x),x=y))**2
    
    #If 1D
    elif np.size(np.shape(F))==1:
        normF=np.sqrt(np.trapz(np.abs(F)**2,x=x))
        normG=np.sqrt(np.trapz(np.abs(G)**2,x=x))
        
        return 1.0/(normF*normG)**2*np.abs(np.trapz(F*np.conjugate(G),x=x))**2

def NRMS(data,estimate):
    '''
    Calculate the normalized Root Mean Squared Error of the measured and
    estimated modulous.

    Parameters
    ----------
    data: ndarray
        Measured data.
    estimate: ndarray
        Reconstructed estimate.

    Returns
    ----------
    nrms: float
        NRMS Error
    mu: float
        scaling parameter in least squares fit.
    '''

    x1,x2=np.double(np.abs(data)),np.double(np.abs(estimate))
    x1=np.double(x1)/(np.max(x1))
    x2=np.double(x2)/(np.max(x2))
    
    
    fit=least_squares(lambda mu,x1,x2: (x1-mu*x2),1.0,args=(x1.reshape(-1),x2.reshape(-1)))

    mu=fit.x[0]
    nrms=np.sqrt(2*fit.cost/np.sum(x1**2))

    return  nrms,mu

def chi2_reduced(data,estimate):
    '''
    Calculate the reduced chi-squared of the intensity.

    Parameters
    ----------
    data: ndarray
        Measured data.
    estimate: ndarray
        Reconstructed estimate.

    Returns
    ----------
    nrms: float
        Reduced chi-squared
    mu: float
        scaling parameter in least squares fit.
    '''
    x1,x2=data,estimate
    #x1=np.double(x1)/np.max(x1)
    #x2=np.double(x2)/np.max(x2)
    fit=least_squares(lambda n,x1,x2: (x1-n*x2)/np.sqrt(n*x2+0.001),
                      x1.max(),args=(x1.reshape(-1),x2.reshape(-1)))
    n=fit.x[0]
    chi2=2*fit.cost/(x1.size-1)
    return  chi2,n

def error_I(data,estimate):
    '''
    Find the error in the intensities.

    Parameters
    ----------
    data: ndarray
        Measured data.
    estimate: ndarray
        Reconstructed estimate.

    Returns
    ----------
    G: float
        Error in the intensities.
    mu: float
        scaling parameter in least squares fit.
    '''
    I1,I2=data,estimate
    #I1=np.double(I1)/np.max(I1)
    #I2=np.double(I2)/np.max(I2)
    fit=least_squares(lambda mu,x,y: (x-mu*y),
                      np.max(np.abs(I1))/np.max(np.abs(I2)),
                      args=(I1.reshape(-1),I2.reshape(-1)))
    mu=fit.x[0]
    #G=np.sqrt(2*fit.cost/np.sum(I1)**2)
    G=np.sqrt(2*fit.cost/(I1.size))
    return  G,mu
    
def error_phi(data,estimate):
    '''
    Find the error in the phase normalized to the amplitude.

    Parameters
    ----------
    data: ndarray
        Measured data.
    estimate: ndarray
        Reconstructed estimate.

    Returns
    ----------
    G: float
        Error in the intensities.
    phi:ndarray 
        
    '''

    E1,E2=data,estimate
    #I1=np.double(I1)/np.max(np.abs(I1))
    #I2=np.double(I2)/np.max(np.abs(I2))
    
    fit=least_squares(lambda p,x,y: np.abs(x)*(np.angle(x)-np.angle(p[0]*np.exp(1j*p[1])*y))
                      ,[1.0,0.0],args=(E1.reshape(-1),E2.reshape(-1)))
    phi=fit.x
    norm=np.sqrt(np.sum(np.abs(E1)**2))
    G=np.sqrt(2*fit.cost)/norm
    return  G,phi

def BhattacharyyaDistance(p,q):
    '''
    Calculate the Bhattacharyya distance measuring the similarity of two probability distributions.

    Parameters
    ----------
    p: ndarray
        1D or 2D array.

    q: ndarray
        1D or 2D array

    Returns
    ----------
    out:int
        Bhattacharyya distance between p and q. 
        
    '''
    return -np.log(np.sum(np.sqrt(p/np.sum(p)*q/np.sum(q))))

def BhattacharyyaCoeff(p,q):
    '''
    Calculate the Bhattacharyya distance measuring the similarity of two probability distributions.

    Parameters
    ----------
    p: ndarray
        1D or 2D array.

    q: ndarray
        1D or 2D array

    Returns
    ----------
    out:int
        Bhattacharyya distance between p and q. 
        
    '''
    return np.sum(np.sqrt(p/np.sum(p)*q/np.sum(q)))

def random_phase(X):
    """
    Retrun a random phase with shape of X.

    Parameters
    ----------
    X: ndarray
        Input array.

    Returns
    ----------
    out:ndarray
        Ouput array of random phase with same shape as input array.

    """
    return np.exp(1j*(-np.pi+np.random.rand(*np.shape(X))*2*np.pi))

def FROG_error(data,estimate):
    '''
    Calculate FROG intensity error between two images. 
    See Trebino Ch. 8 page 160.

    Parameters
    ----------
    data: ndarray
        Measured data.
    estimate: ndarray
        Reconstructed estimate.

    Returns
    ----------

    G: float
        FROG Error
    mu: float
        scaling parameter in least squares fit.

    '''
    x1,x2=data,estimate

    #Normmalize
    x1=np.double(x1)/np.max(x1)
    x2=np.double(x2)/np.max(x2)

    #Find optimal scale parameter
    fit=least_squares(lambda mu,x1,x2: (x1-mu*x2),1,args=(x1.reshape(-1),x2.reshape(-1)))

    mu=fit.x[0]

    G=np.sqrt(2*fit.cost/x1.size)

    return  G,mu