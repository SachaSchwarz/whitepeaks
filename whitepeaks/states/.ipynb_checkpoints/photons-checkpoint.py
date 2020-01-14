'''
###############################################################################
                        Two-Photon States 
###############################################################################

Quantum Optics and Quantum Information Group
Written by Jean-Philippe MacLean: jpmaclean@uwaterloo.ca

Functions to create two-photon correlated Gaussian states.

'''
import numpy as np
from ..fit import *
#from ..sfg import *
from ..optics import *

c=0.299792458 #Speed of light in um/fs or mm/ps

def two_photon_frequency_grid(wi0,ws0,sigma_i,sigma_s,Nsamples=[2**8,2**8],Nsigma=[4.5,4.5]
                                     ,sparseGrid=False):
    '''
    Make a two-photon frequency grid centered on (wi0,ws0) with marginals
    (sigma_i,sigma_s). 

    Nsamples is the number of data points. Nsigma is the size of the grid
    in frequency sigmas.
    '''

    if np.size(Nsamples)==1:
        Ni,Ns=Nsamples,Nsamples
    else: 
        Ni,Ns=Nsamples
        
    if np.size(Nsigma)==1:
        Nsigi,Nsigs=Nsigma,Nsigma
    else:
        Nsigi,Nsigs=Nsigma
    
    #Frequency range
    wir=(Nsigi*np.sqrt(np.log(256))*(sigma_i))
    wsr=(Nsigs*np.sqrt(np.log(256))*(sigma_s))
    
    #Frequency spacing
    dwi=wir/(0.5*Ni)
    dws=wsr/(0.5*Ns)
    
    #Frequency interval
    wi=np.linspace(wi0-wir,wi0+wir-dwi,Ni)
    ws=np.linspace(ws0-wsr,ws0+wsr-dws,Ns)
    
    #Make grid   
    if sparseGrid:
        Wi,Ws=np.meshgrid(wi,ws,sparse=True)
    else:
        Wi,Ws=np.meshgrid(wi,ws)
        
    return Wi,Ws

def gaussian_state(wi0,ws0,sigma_i,sigma_s,rho,Ai=0,As=0,Ap=0,Nsamples=[2**8,2**8],Nsigma=[4.5,4.5],
                  sparseGrid=False):
    '''
    Create a frequency correlated gaussian state.
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

def fft_state_1D(Wi,Ws,F,axis=0):
    '''
    One-dimensional FFT of the input state along specified axis. 
    '''

    dwi=np.diff(Wi)[0,0]
    dws=np.diff(Ws.T)[0,0]
    
    ti=np.linspace(-2.0*np.pi/(2*dwi),2.0*np.pi/(2*dwi),np.shape(Wi)[1]) 
    ts=np.linspace(-2.0*np.pi/(2*dws),2.0*np.pi/(2*dws),np.shape(Ws)[0]) 
    Ti,Ts=np.meshgrid(ti,ts)
   
    f=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(F,axes=axis),axes=[axis]),axes=axis)
    
    #Normalize f to satisfy Parseval's theorem
    f*=np.sqrt(np.sum((np.abs(F)**2))/np.sum(np.abs(f)**2))
    
    if axis==0:
        return Wi,Ts,f
    else:
        return Ti,Ws,f

def fft_state_2D(Wi,Ws,F):
    '''
    FFT the input state. 
    '''

    #Frequency spacing
    dwi=np.diff(Wi)[0,0]
    dws=np.diff(Ws.T)[0,0]
   
    #Create appropriate time intervals for fft.
    ti=np.linspace(-2.0*np.pi/(2*dwi),2.0*np.pi/(2*dwi),np.shape(Wi)[1]) 
    ts=np.linspace(-2.0*np.pi/(2*dws),2.0*np.pi/(2*dws),np.shape(Ws)[0]) 

    #Make grid
    Ti,Ts=np.meshgrid(ti,ts)
   
    #Joint temporal intensity
    f=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(F)))
    
    #Normalize f to satisfy Parseval's theorem
    f*=np.sqrt(np.sum((np.abs(F)**2))/np.sum(np.abs(f)**2))
    
    return Ti,Ts,f

def SPDC_state(wi0,ws0,sigma_i,sigma_s,pump_bandwidth,crystal,L,angle,
               Nsamples=[2**8,2**8],Nsigma=[4.5,4.5],
               Ai=0,As=0,Ap=0,
               tau_p=1000,pump_aBBOangle=0,pump_HWPangle=0,pump_aBBO_length=1e3):

    gamma=0.04822

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
           +0.5*d2ks*(Ws-ws0)**2+0.5*d2ki*(Wi-wi0)**2-0.5*d2kp*(Ws+Wi-wp0)**2)
    phi=np.pi*np.sinc(deltak*L/2.0/np.pi)
    #phi=np.exp(-gamma*(deltak*L)**2) #Approximate phase matching function
    
    #JOINT SPECTRUM
    F=(alpha*B*phi)
    F=F*np.exp(1j*Ai*(Wi-wi0)**2+1j*As*(Ws-ws0)**2+1j*Ap*(Ws-ws0+Wi-wi0)**2)
    F/=np.sqrt(np.max(np.abs(F)**2))

    return Wi,Ws,F

