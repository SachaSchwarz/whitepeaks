'''
###############################################################################
BINNER MODULE
###############################################################################

Routines in this module:

Mappings
--------
> w2f(x)
> dw2f(x,dx)
> freqspace(w0,sigma)
> two_photon_frequency_grid(wi0,ws0,sigma_i,sigma_s)
> padnextpow2(X,Y,Z)
> square_grid(x0,dx,y0,dy,n)
> rect_ycut(X)

FROG binner
-----------
> binner(X,Y,Z,grid)
> FROG_grid(x)

GS binner
---------
> grid2d_data(x,y,z)

'''

import numpy as np

from scipy import interpolate
from ..analytics.fft import *

'''
-------------------------------------------------------------------------------
Mappings
-------------------------------------------------------------------------------
'''
c=0.299792458

def w2f(x,
        dx=0):
     """
     Convert the wavelength to frequency or vice-versa

    Parameters
    ----------
    x:float
        Wavelength or angular frequency. c in units of um/fs or mm/ps
    dx:float,optional
        RMS bandwidth in wavelength or angular frequency.

    Returns
    ----------
    out:float
        Frequency or wavelength depending on the input.
    or
    out:tuple(float,float)
        Frequency or wavelength and bandwidth depending on the input.

     """

     omega=2*np.pi*c/x

     if dx:
         domega=2*np.pi*c/x**2*dx
         return omega,domega

     else: 
         return omega

def dw2f(x,dx,
         error=0):
     """
     Bandwidth conversion from wavelength to frequency. 

    Parameters
    ----------
    x:float
        Wavelength or angular frequency. c in units of um/fs or mm/ps
    dx:float,optional
        RMS bandwidth in wavelength or angular frequency.

    Returns
    ----------
    out:float
        Bandwidth in frequency or wavelength depending on the input.
    or
    out:tuple(float,float)
        Error in bandwidth in frequency or wavelength and bandwidth depending on the input.
     """

     omega=2*np.pi*c/x

     domega=2*np.pi*c/x**2*dx
     if error:
         ddx=error
         ddomega=ddx/dx*domega

         return domega,ddomega
     else: 
         return domega

def freqspace(w0,sigma,
              Nsamples=2**8,Nsigma=4.5):
    '''
    Make a frequency line centered on w0 with standard deviation sigma. 

    Parameters
    ----------
    w0: float 
        Centre angular frequency.
    sigma:float
        Frequency bandwidth in standard deviations. 
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
    '''
    
    #Frequency range
    wr=(Nsigma*np.sqrt(np.log(256))*(sigma))
    
    #Frequency spacing 
    dw=wr/(0.5*Nsamples)
    
    #Frequency interval 
    w=np.linspace(w0-wr,w0+wr-dw,Nsamples)
    
    return w 

def two_photon_frequency_grid(wi0,ws0,sigma_i,sigma_s,
                              Nsamples=[2**8,2**8],Nsigma=[4.5,4.5],sparseGrid=False):
    '''
    Make a two-photon frequency grid centered on (wi0,ws0) with marginals
    (sigma_i,sigma_s). 

    Nsamples is the number of data points. Nsigma is the size of the grid
    in frequency sigmas.

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
    Nsamples:list (float,float)
        Number of sample points along x and y in the grid.
    Nsigma:list (float,float)
        Size of the grid specfied a multiplicative factor of the standard
        deviation along x and y.
    sparseGrid:bool,optional
        Make the state a sparse matrix

    Returns
    ----------
    out:tuple(ndarray,ndarray)
        Wi:ndarray 
            2D meshgrid array of idler frequencies
        Ws:ndarray 
            2D meshgrid array of signal frequencies
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

def padnextpow2(X,Y,Z,
                grid_size=0):
    '''
    Extend X,Y arrays grid keeping the same grid spacing and pad Z with
    zeros so that the size of the grid is the next power of 2 or more.

    Parameters
    ----------
    X: ndarray
        2D x axis meshgrid array
    Y: ndarray
        2D y axis meshgrid array
    Z: ndarray
        2D z axis array
    grid_size: int, optional
        Specify a particular grid size to use.

    Returns
    ----------
    out:tuple(ndarray,ndarray,ndarray)
        Extended X,Y meshgrid with padded array Z.

    '''
    x=X[0,:]
    #print X
    xmin,xmax,dx=x.min(),x.max(),np.diff(x)[0]

    y=Y[:,0]
    #print Y
    ymin,ymax,dy=y.min(),y.max(),np.diff(y)[0]

    #Find next power of 2
    if grid_size:
        np2x=np.log2(grid_size)
        np2y=np.log2(grid_size)
    else:
        np2x=int(np.ceil(np.log2(x.size)))
        np2y=int(np.ceil(np.log2(y.size)))
        #Take largest value
        if np2x>np2y: np2y=np2x
        else: np2x=np2y

    #Extend start and stop for array size 2**nextpow2 
    xpad=int(2**np2x-x.size)
    xmin=xmin-xpad//2*dx
    xmax=xmin+(2**np2x-1)*dx

    ypad=int(2**np2x-y.size)
    ymin=ymin-ypad//2*dy
    ymax=ymin+(2**np2y-1)*dy
    
    #Define array 
    xlin=np.arange(xmin,xmax+0.5*dx,dx)
    ylin=np.arange(ymin,ymax+0.5*dy,dy)

    #INTERPOLATED GRID
    X,Y=np.meshgrid(xlin,ylin)
    
    Z=np.pad(Z,pad_width=((ypad//2,ypad-ypad//2),(xpad//2,xpad-xpad//2)),mode='constant')

    return X,Y,Z

def square_grid(x0,dx,y0,dy,n):
    '''
    Create square grid from center, spacing, and number of points

    Parameters
    ----------
    x0:float 
       x0 grid centre
    dx:float
       Grid spacing along x-axis
    y0:float 
       y0 grid centre
    dy:float
       Grid spacing along y-axis
    n:integer
       Number of points along one axis in the square grid.

    Returns
    ----------
    out:tuple(ndarray,ndarray)
        X:2D meshgrid array
        Y:2D meshgrid array
    '''
    #Create linear spacing
    xlin=np.arange(x0-n//2*dx,x0+n//2*dx,dx)
    ylin=np.arange(y0-n//2*dy,y0+n//2*dy,dy)

    #Make number of points is right 
    xlin,ylin=xlin[0:n],ylin[0:n]

    X,Y=np.meshgrid(xlin,ylin)

    return X,Y

def rect_ycut(X,
              fract=0.5):
    """
    Return a copy of X with elements to the right of the cut zeroed.
    """
    n=np.array(X.shape)
    X[:,np.int(np.floor(fract*n[1])):-1]=0
    return X

'''
-------------------------------------------------------------------------------
FROG binner
-------------------------------------------------------------------------------
'''
def binner(X,Y,Z,grid,method='cubic'):
    '''
    Bin data according to specified grid.

    Parameters
    ----------
    X: ndarray
        2D x axis meshgrid array
    Y: ndarray
        2D y axis meshgrid array
    Z: ndarray
        2D z axis array
    grid: tuple (Xb,Yb)
        Xb,Yb meshgrid to fit data to.
    method: {'cubic','linear','nearest'}
        Interpolation methods. See np.interpolate.griddata.

    Returns
    ----------
    out: tuple (ndarray,ndarray,ndarray)
        (Xb,Yb,Zb) tuple of ndarrays of data on grid. 
    '''
    x=(X.reshape(-1)).copy()
    y=(Y.reshape(-1)).copy()
    z=(Z.reshape(-1)).copy()

    #Specified Grid   
    X,Y=grid
    Z=interpolate.griddata(np.vstack((x,y)).T,z,(X,Y),method=method)

    #Remove NANs and negative numbers
    Z[np.isnan(Z)]=0
    Z[Z<0]=0

    return X,Y,Z

def FROG_grid(x,
              k0=0,axis=0,n=64):
    '''
    Make a grid which satisfies the grid requirements for FROG, i.e.
    number of points is a power of 2 and frequency and time axes are
    Fourier transforms.
    
    Parameters
    ----------
    x:ndarray or tuple (int,int)
        Either an array in one dimension or a tuple (x0,dx)
        specifying the center of the array and the spacing between points.
    k0:float,optional
        Centre frequency or time.
    axis:int,optional
        Use axis=0 to define the grid in terms of time and axis=1 to define it
        in terms of frequency.
    n: int,optional
        Number of points along each axis. Must be a power of 2.

    Returns
    ----------
    out: tuple (ndarray,ndarray)
        (X,Y) FROG meshgrid

    '''
    #Check if x is an array.
    if isinstance(x,np.ndarray):

        xmin,xmax,dx=x.min(),x.max(),np.diff(x)[0]

        if n>x.size:
            #Extend grid to the specified number of points
            xpad=int(n-x.size)

            xmin=xmin-xpad//2*dx
            xmax=xmin+(n-1)*dx

            xlin=np.arange(xmin,xmax+0.5*dx,dx)

        elif n<x.size:
            #Reduce the grid to the specified number of points
            start=(x.size-n)//2
            xlin=x[start:start+n]
        else:
            xlin=x

    #Check if x is a tuple x=(x0,dx) 
    elif isinstance(x,tuple):
        x0,dx=x

        #Create line
        xlin=np.arange(x0-n//2*dx,x0+n//2*dx,dx)
        #Make number of points right 
        xlin=xlin[0:n]
    else:
        print('Argument x not an array or a tuple.') 

    #FFT line
    ylin=fft_axis(xlin)+k0

    if axis==0:
        X,Y=np.meshgrid(xlin,ylin)
    elif axis==1:
        X,Y=np.meshgrid(ylin,xlin)

    return X,Y

'''
-------------------------------------------------------------------------------
GS binner
-------------------------------------------------------------------------------
'''
def grid2d_data(x,y,z,
                grid='data',grid_size=0,method='nearest'):
    '''
    Take a list of data points (x,y,z) such that 
    x=[x1,x1,x1,x2,x2,x2...]
    y=[y1,y2,y3,y1,y2,y3...]
    and grid the data using meshgrid. 

    Different grid options are provided. 

    Parameters
    ----------
    x:ndarray
        x-coordinates of the grid.
    y:ndarray
        y-coordinates of the grid.
    z:ndarray
        z-coordinates of the grid.
    grid:{tuple,list,'data','double_time','nextpow2'},optional
    Grid method
        tuple: Specify a tuple of 2D meshgrid arrays (X,Y)
        list: Specify list of max/min [xmin,xmax,ymin,ymax] keeping same
        spacing between points as in data.
        data: Use grid based on measured x,y points.
        double_time: Use grid max/min based on x,y points with the number of point
        inside the grid doubled.
        nextpow2: Increase the grid based on x,y points to the next power
        of 2 keeping the same spacing between points.
    grid_size:{},optional
    method: {'cubic','linear','nearest'}
        Interpolation methods. See np.interpolate.griddata.

    Returns
    ----------
    out:tuple(ndarray,ndarray,ndarray)
        X:ndarray
            2D meshgrid array
        Y:ndarray
            2D meshgrid array
        Z:ndarray
            z-values at (X,Y)


    '''

    n=np.where(np.abs(np.diff(x))>0)
    #nx,ny=n[0].size,n[0][0]+1
    ny=(n[0][0]+1)
    nx=(x.size//ny)
    #print(nx,ny)
    x,y,z=x[0:nx*ny],y[0:nx*ny],z[0:nx*ny]

    xmin,xmax=np.min(x),np.max(x)
    ymin,ymax=np.min(y),np.max(y)

    dx=(x.max()-x.min())/(nx-1)
    dy=(y.max()-y.min())/(ny-1)

    #dx=np.abs(np.diff(x)[np.where(np.diff(x)>0)].mean())
    #dy=np.abs(np.diff(y)[np.where(np.diff(y)<0)].mean())

    if isinstance(grid,tuple):
    #Specified Grid   
        X,Y=grid
        Z=interpolate.griddata(np.vstack((x,y)).T,z,(X,Y),method=method)

    elif isinstance(grid,list):
        #Grid based on data but extend to specified limits
        if len(grid)==4:
            xmin,xmax,ymin,ymax=grid

            xmin=xmin-(xmin-x.min())%dx
            xmax=xmax+(dx-(xmax-x.max())%dx)

            ymin=ymin-(ymin-y.min())%dy
            ymax=ymax+(dy-(ymax-y.max())%dy)

            xlin=np.arange(xmin,xmax+dx,dx)
            ylin=np.arange(ymin,ymax+dy,dy)
        else: 
            print('Please specify bounds [xmin,xmax,ymin,ymax] for limits')

       # xlin=np.arange(int((x.min()-xmin)/dx)*dx+x.min(),int((x.max()-xmax)/dx)*dx+x.max(),dx)
       # ylin=np.arange(int((y.min()-ymin)/dy)*dy+y.min(),int((y.max()-ymax)/dy)*dy+y.max(),dy)

        X,Y=np.meshgrid(xlin,ylin)
        Z=interpolate.griddata(np.vstack((x,y)).T,z,(X,Y),method=method)

    elif grid=='nextpow2':
        '''
        Extend grid keeping the same grid spacing so that the size of the
        grid is the next power of 2.
        '''

        #Find next power of 2
        if grid_size:
            np2x=np.log2(grid_size)
            np2y=np.log2(grid_size)
        else:
            np2x=np.ceil(np.log2(nx))
            np2y=np.ceil(np.log2(ny))

        #Extend start and stop for array size 2**nextpow2 
        xmin=xmin-((2**np2x-nx)//2)*dx
        xmax=xmin+(2**np2x-1)*dx

        ymin=ymin-((2**np2x-ny)//2)*dy
        ymax=ymin+(2**np2y-1)*dy
        
        #Define array 
        xlin=np.arange(xmin,xmax+0.5*dx,dx)
        ylin=np.arange(ymin,ymax+0.5*dy,dy)

        #INTERPOLATED GRID
        X,Y=np.meshgrid(xlin,ylin)
        Z=interpolate.griddata(np.vstack((x,y)).T,z,(X,Y),method=method)
        
    elif grid=='data':
        #Grid based on available data
        xlin=np.linspace(xmin,xmax,nx)
        ylin=np.linspace(ymin,ymax,ny)

        #INTERPOLATED GRID
        X,Y=np.meshgrid(xlin,ylin)
        Z=interpolate.griddata(np.vstack((x,y)).T,z,(X,Y),method=method)

    elif grid=='double_time':
        xlin=np.linspace(xmin,xmax,2*nx-1)
        ylin=np.linspace(ymin,ymax,2*ny-1)

        #INTERPOLATED GRID
        X,Y=np.meshgrid(xlin,ylin)
        Z=interpolate.griddata(np.vstack((x,y)).T,z,(X,Y),method=method)

    return X,Y,Z
