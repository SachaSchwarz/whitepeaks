'''
###############################################################################
POST-PROCESSING MODULE
###############################################################################

Routines in this module:

> top_hat(N,rho)
> low_pass_filter(h)
> corner_suprresion(Z,d)
> corner_subtract(Z,l)
> background_subtract_array(Z,number_of_columns)
> background_subtract_treshold(A,threshold)
> fftdeconvolve(g,h)

'''

import numpy as np

def top_hat(N,rho):
    '''
    Create top hat mask. 
    
    Parameters
    ----------
    N:  tuple (int,int) 
        Desired shape (x,y) of input array.

    rho:int
        Radius between (0,1) of top hat mask to apply in Fourier space.

    Returns
    ----------
    out:ndarray
        Array of shape N=(nx,ny) with a top hat function of dimension of
        radius rho n/2, where all values outside this radius are set to 0. 
    '''

    nx,ny=N
    X=np.ones((nx,ny))
    i0,j0=nx/2,ny/2
    for i in range(nx):
        for j in range(ny):
            if np.sqrt((i-i0)**2+(j-j0)**2)>rho*np.sqrt(X.size)/2:
                X[i,j]=0
    return X

def low_pass_filter(h,
                    npad=0,rho=1,epsilon=0):
    '''
    Apply low pass filter with a top hat mask of radius rho to 2D array h.  

    See Trebino Ch9 page 188-189.

    Parameters
    ----------
    h:  ndarray
        2D input array.

    npad:int,optional
        Number of data points to pad with.

    rho:int,optional
        Radius between (0,1) of top hat mask to apply in Fourier space.

    epsilon: float, optional
        Frequency amplitudes below epsilon (percentage of maximum) are set
        to 0. Works as a low pass filter if the input array has one
        central feature which decays towards the outside. 

    Returns
    ----------
    out:ndarray
        2D array with low pass filter applied.

    '''

    #Check whether a specific padding has been specified.
    if not(np.bool(npad)):
        hm=np.sum(h,axis=1)
        npad=np.where(hm/np.max(hm)<0.001)[0].size//2
        #print npad

    h=np.pad(h,npad,'constant')
    
    H=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(h)))

    if epsilon!=0:
        #Remove small frequency amplitudes. 
        H[np.abs(H)/np.max(np.abs(H))<epsilon]=0
        
    #Apply a top hat filter with radius rho*N to remove high
    #frequency components. 
    H*=top_hat(np.shape(H),rho)
    
    f=np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(H))))

    return f[npad:-npad,npad:-npad]

def corner_suprresion(Z,d):
    '''
    Apply super gaussian to remove noise in the corners of the
    spectrogram.

    Trebino Ch9 Eq. 9.5 page 188

    Parameters
    ----------
    Z: 2D array
    d: FWHM in pixels of super gaussian 

    Returns
    ----------
    out: ndarray
        input array multiplied by super-gaussian.

    '''

    #Array size
    nx,ny=np.shape(Z.T)

    #Centre
    x0,y0=nx//2,ny//2

    x=np.arange(nx)
    y=np.arange(ny)
    X,Y=np.meshgrid(x,y)

    supergauss=np.exp(-16*np.log(2)*((X-x0)**2+(Y-y0)**2)**2/d**4)

    return Z*supergauss

def corner_subtract(Z,l):
    '''
    Subtract the average value of the four nxn corners from the array. 
    Parameters
    ----------
    Z: ndarray
        2D input array
    l: int
        Length of the sides of the square corners.

    Returns
    ----------

    '''
    corners=np.concatenate((Z[0:l,0:l],Z[-(l+1):-1,-(l+1):-1],Z[-(l+1):-1,0:l],Z[0:l,-(l+1):-1]))
    Z=Z-corners.mean()
    Z[np.where(Z<0)]=0
    return Z

def background_subtract_array(Z,number_of_columns,
                              nstd=2):
    '''
    Subtract the background by subtracting from the entire array the
    average pixel values at the edges of the spectrogram.

    Parameters
    ----------
    Z: ndarray

    number_of_columns: int 
        Number of columns of noise on edges of array to average. 

    nstd:int,optional
        Values below the average noise + nstd*standard devitaion of
        the noise are set to 0.

    Returns
    ----------
    out: ndarray
        Background subtracted array. 
    '''
    nc=number_of_columns

    #Background
    Zcat=np.concatenate((Z[:,-nc:],Z[:,:nc]),axis=1)
    Zmean=np.mean(Zcat,axis=1)
    Zstd=np.std(Zcat,axis=1)

    Zbkgd=np.ones(np.shape(Z))
    set_to_zero=np.ones(np.shape(Z))
    
    for i in range(np.shape(Zbkgd)[1]):
        #Make bacground columns the mean
        Zbkgd[:,i]=Zmean
        #Calculate truth table for background subtraction
        set_to_zero[:,i]=Z[:,i]-Zmean<nstd*Zstd

    Z=Z-Zbkgd

    Z[np.where(set_to_zero)]=0

    return Z

def background_subtract_treshold(A,threshold):
    '''
    Set signal to zero up to certain threshold.
    
    Parameters
    ----------
    A: ndarray
    
    threshold: float
        Percentage of maximal amplitude value

    Returns
    ----------
    A: ndarray
        Background subtracted array

    '''
    #
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            if A[i,j]<np.max(A)*threshold:
                A[i,j] = 0
                
    return A

def fftdeconvolve(g,h,
                  npad=0,method='wiener',alpha=0,low_pass=0,cut_off=0,epsilon=0):
    '''
    FFT deconvolve g with h with Wiener or inverse filtering

    Parameters
    ----------
    g: ndarray
        2D array to be deconvolved.
    h: ndarray
        2D filter function.
    npad:integer,optional
        Pad array before apply the fft. If npad=0, the padding is
        calculated automatically.
    method:{'wiener','inverse_filter'},optional
        Apply a Wiener filter by specifying the weight alpha.
        Apply an inverse filter by removing all values below cut_off in
        percentage.
    alpha:float,optional
        Weighting to apply during deconvolution with Wiener filter as a
        percentage of maximum value of the filter.
    cut_off:float,optional
        Cut off value in percentage of maximum below which removes all
        small k vectors in the filter function. 
    low_pass:float,optional
        During deconvolution, apply a top hat filter with radius
        low_pass*N to remove high frequency components.
    epsilon:float,optional
        Remove small frequency amplitudes below epsilon in percentage.
        Only works as a low pass filter if it is a central feature which
        decays towards the outside.

    Returns
    ----------
    f: ndarray
        Deconvolved 2D array.

    '''
    #h=h/np.max(h)
    #g=g/np.max(g)
    
    #Check whether a specific padding has been specified.
    if not(np.bool(npad)):
        hm=np.sum(h,axis=1)
        npad=np.where(hm/np.max(hm)<0.001)[0].size//2
        #print npad
    
    
    h=np.pad(h,npad,'constant')
    g=np.pad(g,npad,'constant')
    
    H=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(h)))
    G=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g)))
    #N=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(noise)))

    #SNR=np.abs(H/N)**2
    
    #Define filter
    if method=='wiener':

        if np.isscalar(alpha):

            alpha*=np.max(np.abs(H))**2

        W=np.conjugate(H)/(np.abs(H)**2+alpha)

    elif method=='inverse_filter':

        W=np.conjugate(H)/(np.abs(H)**2+0.0001*np.max(np.abs(H)**2)) 

        W[np.abs(H)**2/np.max(np.abs(H)**2)<cut_off]=0

    else: 
        print('Please specify deconvolution method')

    if epsilon!=0:
        #Remove small frequency amplitudes. Only works as a low pass filter
        #if it is a central feature which decays towards the outside.
        W[np.abs(G)/np.max(np.abs(G))<epsilon]=0
        
    if low_pass!=0:
        #Apply a top hat filter with radius low_pass*N to remove high
        #frequency components. 
        W*=top_hat(np.shape(W),low_pass)
    
    F=G*W
    #U=H*np.conjugate(G)/(np.abs(G)**2)#*(H/(H+N+0.01))
    f=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(F)))
    
    return f[npad:-npad,npad:-npad]
