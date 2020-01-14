'''
###############################################################################
PHOTON STATES MODULE
###############################################################################

This module contains the following functions:

> gaussian_state(wi0,ws0,sigma_i,sigma_s,rho)
> SPDC_state(wi0,ws0,sigma_i,sigma_s,pump_bandwidth,crystal,L,angle)

'''
import numpy as np

from ..analytics import *

from .xstal import *
from .pulses import *
from .waveplates import *

c=0.299792458 #Speed of light in um/fs or mm/ps

def gaussian_state(wi0,ws0,sigma_i,sigma_s,rho,
                   Ai=0,As=0,Ap=0,Nsamples=[2**8,2**8],Nsigma=[4.5,4.5],sparseGrid=False):
    '''
    Create a frequency correlated 2D Gaussian state for idler and signal
    photon pairs.

    Parameters
    ----------
    wi0: float 
        Idler centre frequency.
    ws0: float 
        Signal centre frequency.
    sigma_i:float
        Idler frequency bandwidth standard deviation. 
    sigma_s:float
        Signal frequency bandwidth standard deviation. 
    rho:float
        Statistical correlation between the frequencies of the signal and
        idler pair.
    Ai:float,optional
        Chirp parameter on the idler photon.
    As:float,optional
        Chirp parameter on the signal photon.
    Ap:float,optional
        Chirp parameter on the pump.
    Nsamples:list (float,float)
        Number of sample points along x and y in the grid.
    Nsigma:list (float,float)
        Size of the grid specfied a multiplicative factor of the standard
        deviation along x and y.
    sparseGrid:bool,optional
        Make the state a sparse matrix

    Returns
    ----------
    out:tuple(ndarray,ndarray,ndarray)
        Wi:ndarray 
            2D meshgrid array of idler frequencies
        Ws:ndarray 
            2D meshgrid array of signal frequencies
        F:ndarray 
            2D meshgrid arry of frequency-frequency amplitudes.
    '''
    #Create grid
    if sparseGrid:
        Wi,Ws=two_photon_frequency_grid(wi0,ws0,sigma_i,sigma_s,Nsamples=Nsamples,Nsigma=Nsigma,
                                                   sparseGrid=True)
    else:
        Wi,Ws=two_photon_frequency_grid(wi0,ws0,sigma_i,sigma_s,Nsamples=Nsamples,Nsigma=Nsigma)
    
    #Joint spectral amplitude
    F=1.0/(np.sqrt(2*np.pi*sigma_i*sigma_s)*(1-rho**2)**0.25)*(
    gauss2d([Wi,Ws],1,wi0,ws0,np.sqrt(2)*sigma_i,np.sqrt(2)*sigma_s,rho,0))
    
    #Add dispersion
    F=F*np.exp(1j*Ai*(Wi-wi0)**2+1j*As*(Ws-ws0)**2+1j*Ap*(Ws-ws0+Wi-wi0)**2)

    #Normalize
    F/=np.sqrt(np.max(np.abs(F)**2))

    return Wi,Ws,F

def SPDC_state(wi0,ws0,sigma_i,sigma_s,pump_bandwidth,crystal,L,angle,
               Nsamples=[2**8,2**8],Nsigma=[4.5,4.5],
               Ai=0,As=0,Ap=0,
               pump_aBBOangle=0,pump_HWPangle=0,pump_aBBO_length=0):
    '''
    Create a frequency correlated 2D SPDC state for idler and signal
    photon pairs.

    Parameters
    ----------
    wi0: float 
        Idler angular centre frequency.
    ws0: float 
        Signal angular centre frequency.
    sigma_i:float
        Idler frequency bandwidth standard deviation. 
    sigma_s:float
        Signal frequency bandwidth standard deviation. 
    pump_bandwidth:float
        Frequency bandwidth of the pump.
    crystal:class
        crystal class for BBO, BiBO etc.
    L:float
        Length of the crystal.
    angle:float
    Ai:float,optional
        Chirp parameter on the idler photon.
    As:float,optional
        Chirp parameter on the signal photon.
    Ap:float,optional
        Chirp parameter on the pump.
    Nsamples:list (float,float)
        Number of sample points along x and y in the grid.
    Nsigma:list (float,float)
        Size of the grid specfied a multiplicative factor of the standard
        deviation along x and y.
    pump_aBBOangle:float,optional
    pump_HWPangle:float,optional
    pump_aBBO_length:float,optional


    Returns
    ----------
    out:tuple(ndarray,ndarray,ndarray)
        Wi:ndarray 
            2D meshgrid array of idler frequencies
        Ws:ndarray 
            2D meshgrid array of signal frequencies
        F:ndarray 
            2D meshgrid arry of frequency-frequency amplitudes.
    '''

    Wi,Ws=two_photon_frequency_grid(wi0,ws0,sigma_i,sigma_s,Nsamples=Nsamples,Nsigma=Nsigma)
    wp0=ws0+wi0

    lambda_s0=2*np.pi*c/ws0
    lambda_i0=2*np.pi*c/wi0
    lambda_p0=2*np.pi*c/wp0

    #Pump bandwidth (rms)
    dwp=pump_bandwidth#/(2*np.sqrt(2*np.log(2)))
    alpha0=np.exp(-(Ws+Wi-wp0)**2/(2*dwp**2))
    A=(np.cos(pump_aBBOangle-np.pi/4)+np.sin(pump_aBBOangle-np.pi/4))/2
    B=(np.cos(pump_aBBOangle-np.pi/4)-np.sin(pump_aBBOangle-np.pi/4))/2
    cr_aBBO=crystal.alphaBBO()
    tau_p=pump_aBBO_length/c*np.abs(cr_aBBO.sellemeir(lambda_p0,cr_aBBO.ne)-cr_aBBO.sellemeir(lambda_p0,cr_aBBO.no))
    phase_AB=np.exp(1j*(tau_p*(Ws+Wi-wp0)+pump_HWPangle))
    alpha=alpha0*(A+B*phase_AB) 
    
    #Filter bandwidths approximated by gaussians (rms)
    dPis=sigma_s#/(2*np.sqrt(2*np.log(2)))
    dPii=sigma_i#/(2*np.sqrt(2*np.log(2)))

    B=np.exp(-(Ws-ws0)**2/(4*dPis**2))*np.exp(-(Wi-wi0)**2/(4*dPii**2)) 
    
    #Crystal Axes
    cr=crystal.BiBO()
    extra_ordinary=[angle,cr.ny,cr.nz] #Extra-ordinary
    ordinary=cr.nx #Ordinary 
    
    #GV and GVD
    dks,d2ks=1.0/GV(lambda_s0,cr.n_e,extra_ordinary),GVD(lambda_s0,cr.n_e,extra_ordinary) #signal
    dki,d2ki=1.0/GV(lambda_i0,cr.n_e,extra_ordinary),GVD(lambda_i0,cr.n_e,extra_ordinary) #idler
    
    #GV and GVD
    dkp,d2kp=1.0/GV(lambda_p0,cr.sellemeir,ordinary),GVD(lambda_p0,cr.sellemeir,ordinary) #pump
    
    #Phasematching
    deltak=(dks*(Ws-ws0)+dki*(Wi-wi0)-dkp*(Wi+Ws-wp0)
           +0.5*d2ks*(Ws-ws0)**2+0.5*d2ki*(Wi-wi0)**2-0.5*d2kp*(Ws+Wi-wp0)**2
           )
    
    phi=np.sinc(1/np.pi*deltak*L/2)*np.pi*np.exp(-1j*deltak*L/2)
    #gamma=0.04822
    #phi=np.exp(-gamma*(deltak*L)**2)#*np.pi*np.exp(-1j*deltak*L/2) #Approximate phase matching function
    
    #JOINT SPECTRUM
    F=(alpha*B*phi)
    F=F*np.exp(1j*Ai*(Wi-wi0)**2+1j*As*(Ws-ws0)**2+1j*Ap*(Ws-ws0+Wi-wi0)**2)
    F/=np.sqrt(np.max(np.abs(F)**2))

    return Wi,Ws,F
