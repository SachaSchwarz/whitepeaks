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

def ifft_pulse(w,Ew,
               x0=0):
    '''
    One-dimensional FFT of the input pulse. From frequency to time.
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
    One-dimensional FFT of the input pulse. From time to frequency. 
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
    "Input grid in one unit and define grid in fourier unit"
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