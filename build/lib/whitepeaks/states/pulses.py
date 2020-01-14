'''
###############################################################################
PULSES MODULE
###############################################################################

This module contains the following functions:

> gaussian_pulse(w0,sigma)
> SPDC_state(wi0,ws0,sigma_i,sigma_s,pump_bandwidth,crystal,L,angle)

'''

import numpy as np

from ..analytics import * 
from ..interface.binner import *

c=0.299792458 #Speed of light in um/fs or mm/ps

def gaussian_pulse(w0,sigma,
                   A=0,Nsamples=2**8,Nsigma=4.5):
    '''
    Create a gaussian pulse with frequency centered on (w0), an intensity
    bandwidth (sigma), and a quadratic spectral phase (A) on a pulse. 

    Parameters
    ----------
    w0: float 
       Centre angular frequency.
    sigma:float
        Frequency bandwidth standard deviation. 
    A:float,optional
        Chirp parameter on the pulse.
    Nsamples:float
        Number of sample points in the array.
    Nsigma:float
        Size of the grid specfied a multiplicative factor of the standard
        deviation.

    Returns
    ----------
    out:tuple(ndarray,ndarray,ndarray)
        w:ndarray 
            1D array of frequencies
        E:ndarray 
            1D array of frequency-frequency amplitudes.
    '''


    #Create grid
    w=freqspace(w0,sigma,Nsamples=Nsamples,Nsigma=Nsigma)
    
    #Spectral amplitude of the field
    E=1.0/np.sqrt(2*np.pi*sigma)*(gauss(w,1,w0,np.sqrt(2)*sigma,0))
    
    #Add dispersion
    E=E*np.exp(1j*A*(w-w0)**2)

    #Normalize
    E/=np.sqrt(np.max(np.abs(E)**2))

    return w,E
