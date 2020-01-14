'''
###############################################################################
FFT MODULE
###############################################################################

This module contains the following functions:
> fft_state_1D(Wi,Ws,F)
> fft_state_2D(Wi,Ws,F)
> ifft_pulse(w,Ew)
> fft_pulse(t,Et)
> fft_axis(x)
> fft_grid(X,Y)


'''

import numpy as np

def fft_state_1D(Wi,Ws,F,
                 axis=0):
    '''
    One-dimensional FFT of the input state along specified axis. 

    Parameters
    ----------
    Wi: ndarray 
        2D meshgrid of Idler frequencies
    Ws: ndarray 
        2D array of Signal frequencies.     
    F: ndarray 
        2D array of Frequency-Frequency amplitudes.     
    axis:{'0','1'} 
        Specify which axis to apply FFT.

    Returns
    ---------- 
    out: ndarray
    Tuple of arrays in Fourier space.
        {Wi,Ti}: ndarray 
        {Ws,Ts}: ndarray 
        F: ndarray
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
    Two-dimensional FFT. 

    Parameters
    ----------
    Wi: ndarray 
        2D meshgrid of Idler frequencies
    Ws: ndarray 
        2D array of Signal frequencies.     
    F: ndarray 
        2D array of Frequency-Frequency amplitudes.     

    Returns
    ---------- 
    out: ndarray
    Tuple of arrays in Fourier space.
        Ti: ndarray 
            2D meshgrid of Idler times. 
        Ts: ndarray 
            2D meshgrid of Signal times. 
        F: ndarray
            2D array of Time-Time amplitudes.     
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

def ifft_pulse(w,Ew,
               x0=0):
    '''
    One-dimensional iFFT of the input pulse. From frequency to time.

    Parameters
    ----------
    w: ndarray 
        1D meshgrid of frequencies
    Ew: ndarray 
        1D array of frequency amplitudes.     

    Returns
    ---------- 
    out: ndarray
    Tuple of arrays in Fourier space.
        t: ndarray 
            1D array of times. 
        Et: ndarray
            1D array of time amplitudes.     
    '''
    dw=np.diff(w)[0]
    
    t=x0+np.linspace(-2.0*np.pi/(2*dw),2.0*np.pi/(2*dw),np.size(w)) 
   
    Et=np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ew)))
    
    #Normalize f to satisfy Parseval's theorem
    Et*=np.sqrt(np.sum((np.abs(Ew)**2))/np.sum(np.abs(Et)**2))
    
    return t,Et

def fft_pulse(t,Et,
              x0=0):
    '''
    One-dimensional FFT of the input pulse. From frequency to time.

    Parameters
    ----------
    w: ndarray 
        1D array of frequencies
    Ew: ndarray 
        1D array of frequency amplitudes.     

    Returns
    ---------- 
    out: ndarray
    Tuple of arrays in Fourier space.
        t: ndarray 
            1D array of times. 
        Et: ndarray
            1D array of time amplitudes.     
    '''
    dt=np.diff(t)[0]
    
    w=x0+np.linspace(-2.0*np.pi/(2*dt),2.0*np.pi/(2*dt),np.size(t)) 
   
    Ew=np.fft.fftshift(np.fft.fft(np.fft.fftshift(Et)))
    
    #Normalize f to satisfy Parseval's theorem
    Ew*=np.sqrt(np.sum((np.abs(Et)**2))/np.sum(np.abs(Ew)**2))
    
    return w,Ew

def fft_axis(x):
    '''
    Input 1D array and define array in Fourier space.

    Parameters
    ----------
    x: ndarray
        Should be an array of equally spaced points.

    Returns
    ----------
    out: ndarray
        Array in Fourier space.

    '''
    dx=np.diff(x)[0]

    kx=np.linspace(-2.0*np.pi/(2*dx),2.0*np.pi/(2*dx),x.size) 

    return kx

def fft_grid(X,Y,
             axis='both'):
    '''
    Input grid in one unit and define grid in fourier unit"

    Parameters
    ----------
    X: ndarray 
        2D meshgrid 
    Y: ndarray 
        2D meshgrid     
    axis:{'both','0','1'}
        axis or axes to Fourier Transform
        
    Returns
    ----------
    out: ndarray
        Meshgrid or 2D arrays in Fourier space.
    '''

    x=X[0,:]
    y=Y[:,0]
    dx=np.diff(X)[0,0]
    dy=np.diff(Y.T)[0,0]
    
    kx=np.linspace(-2.0*np.pi/(2*dx),2.0*np.pi/(2*dx),np.shape(X)[1]) 
    ky=np.linspace(-2.0*np.pi/(2*dy),2.0*np.pi/(2*dy),np.shape(Y)[0]) 

    if axis=='both':
        Kx,Ky=np.meshgrid(kx,ky)
    elif axis==0:
        Kx,Ky=np.meshgrid(x,ky)
    elif axis==1:
        Kx,Ky=np.meshgrid(kx,y)

    return Kx,Ky
