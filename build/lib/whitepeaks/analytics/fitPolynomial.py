'''
###############################################################################
FIT POLYNOMIAL MODULE
###############################################################################

This module contains the following functions:

Polynomial fits
---------------
> fit_poly(x,y)
> fit_polynomial(data)
> polyfit2d(x, y, z, deg)
> polyfit2sep(x, y, z)

Polynomial phase fits
---------------------
> fit_phase(w,Ew)
> fit_marginal_phase(X,Y,Z)
> polyfit2phase(Wi,Ws,Fk)
> weighted_polyfit2phase(Wi,Ws,Fk)

'''

import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, least_squares
from skimage.restoration import unwrap_phase

from .functions import *
from .stats import *
from .fitGaussian import *

'''
-------------------------------------------------------------------------------
Polynomial fits
-------------------------------------------------------------------------------
'''
def fit_poly(x,y,
             p0=None,fit='poly3'):
    '''
    Fit the spectral phase of a complex field Ew to a polynomial of degree
    2 or 3 using scipy.optimize.curve_fit method.
    Parameters
    ----------
    x:ndarray
    y:ndarray
    p0: ndarray,optional
        Initial guess for fit parameters.
    fit: {'poly3','poly2'}, optional
        Specify the degree of the polynomial to fit to.

    Returns
    ----------
    out:ndarray
        Optimal fit parameters.
    '''

    if fit=='poly3':
        poly=poly3

    elif fit=='poly2':
        poly=poly2
        
    elif fit=='poly1':
        poly=poly1

    popt,_=curve_fit(poly,x,y,p0=p0)
   
    return popt

def fit_polynomial(data,
                   p0=None):
    '''
    !!!
    I don't know that this is used anywhere. Should we delete this
    function?
    !!!
    '''

    n=np.shape(data)[0]

    if n==2:
        x,y=data
        res=least_squares(poly_cost,p0,args=(x,y))

        return res

    elif n==3:
        X,Y,Z=data

        res=least_squares(poly2D_cost,p0,args=(X.reshape(-1),Y.reshape(-1),Z.reshape(-1)))

        return res

def polyfit2d(x, y, z, deg):
    '''
    Least-squares fit of a 2D polynomial to data. Uses Vandermonde
    matrices. 

    Return the coefficients of a polynomial of degree deg that is the
    least squares fit to the data values z given at points (x,y). Similar
    to numpy.polynomial.polynomial.polyfit but for 2D polynomials.
   
    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample (data) points (x[i], y[i], z[i]).
    y : array_like, shape (M,) 
        y-coordinates of the M sample (data) points (x[i], y[i], z[i]).
    z:  array_like, shape (M,) 
        z-coordinates of the sample (data) points (x[i], y[i], z[i]). 
    deg : 1-D array_like
        Degree(s) of the fitting polynomials. 

    Returns
    ----------
    coef : ndarray, shape (deg[0] + 1, deg[1] +1) 
        Polynomial coefficients ordered from low to high. 

    '''
    #DATA
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    #Degrees of the polynomial
    deg = np.asarray(deg)

    vander = poly.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    z = z.reshape((vander.shape[0],))

    c, r, rank, s = np.linalg.lstsq(vander, z,rcond=None)
    
    return c.reshape(deg+1),r,rank,s

def polyfit2sep(x,y,z,
                deg=3):
    '''
    Least-squares fit of a 2D separable polynomial. Fit to a specific
    polynomial.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample (data) points (x[i], y[i], z[i]).
    y : array_like, shape (M,) 
        y-coordinates of the M sample (data) points (x[i], y[i], z[i]).
    z:  array_like, shape (M,) 
        z-coordinates of the sample (data) points (x[i], y[i], z[i]).

    Returns
    ----------
    coef : ndarray, shape (deg[0] + 1, deg[1] +1) 
        Polynomial coefficients ordered from low to high. Cross-terms are
        all set to 0.
    '''

    p=np.zeros((4,4))
    B = z
    #A = np.array([x*0+1, x, x**2, x**3, y ,y**2, y**3, x*y]).T

    #Deg 2
    if deg==2:
        A = np.array([x*0+1, x, x**2, y ,y**2]).T
        c, r, rank, s = np.linalg.lstsq(A, B,rcond=None)
        #Create polynomial
        p[:,0]=[c[0],c[1],c[2],0]
        p[0,:]=[c[0],c[3],c[4],0]

    #Deg 3
    elif deg==3:
        A = np.array([x*0+1, x, x**2, x**3, y ,y**2, y**3]).T
        c, r, rank, s = np.linalg.lstsq(A, B,rcond=None)
        #Create polynomial
        p[:,0]=[c[0],c[1],c[2],c[3]]
        p[0,:]=[c[0],c[4],c[5],c[6]]

    return p, r, rank, s

'''
-------------------------------------------------------------------------------
Polynomial phase fits
-------------------------------------------------------------------------------
'''
def fit_phase(w,Ew,
              fit='poly',p0=None,deg=3,cutoff=None,xlim=[],plot=False):
    '''
    Fit the spectral phase of a complex field Ew to a polynomial of the
    specified degree.

    Parameters
    ----------
    w: ndarray
        Frequency 1D array.
    Ew: complex ndarray
        Complex field amplitude.
    fit:{'poly','poly3','poly2'}
        Polynomial fitting function.
    cutoff: float, optional
        Cutoff for fit in percentage of field intensity.
    xlim: [xmin,xmax], optional
        Bounds for fitting.

    Returns
    ----------
    out:ndarray
        Array of optimal fit parameters.
    '''

    phase=np.unwrap(np.angle(Ew))
    
    if ((cutoff!=None) & (len(xlim)!=0)):
        xmin,xmax=xlim
        roi=np.where((np.abs(Ew)**2/np.max(np.abs(Ew)**2)>cutoff) &(w>xmin) & (w<xmax))
    elif cutoff!=None:
        roi=np.where(np.abs(Ew)**2/np.max(np.abs(Ew)**2)>cutoff)
    elif len(xlim)!=0:
        xmin,xmax=xlim
        roi=np.where((w>xmin) & (w<xmax))
    else:
        roi=np.where(np.abs(Ew)**2)

    if fit=='poly3':
        popt,_=curve_fit(poly3,w[roi],phase[roi],p0=p0)

    elif fit=='poly2':
        popt,_=curve_fit(poly2,w[roi],phase[roi],p0=p0)

    elif fit=='poly':
        fitg=fit_gaussian((w,np.abs(Ew)**2))
        w0=fitg.x[1]
        popt=poly.polyfit(w[roi]-w0,phase[roi],deg)
   
    if plot==True:
        fig,ax=plt.subplots(1,1,figsize=(5,4))

        ax.plot(w,np.abs(Ew)**2)
        ax2=ax.twinx()
        ax2.plot(w,np.unwrap(np.angle(Ew)),'--')

        if fit!='poly':
            ax2.plot(w[roi],fpoly(w[roi],*popt))
        else:
            ax2.plot(w[roi],poly.polyval(w[roi]-w0,popt))

        ax.set_title(r'$E(\omega)$')
        ax.set_xlabel(r'Frequency (fs$^-1$)')
        ax.set_ylabel(r'Amplitude')
        ax2.set_ylabel(r'Phase')

        plt.tight_layout()
        plt.show()
        
        print('Optimal parameters: ',popt)

    return popt

def fit_marginal_phase(X,Y,Z,
                       index=[],tol=0.1,p0=[None,None],plots=True,fits=True):
    '''
    Fit the marginal spectral phase of a 2D complex field Z to a
    third-order polynomial using the scipy.optimize.curve_fit method. 

    Parameters
    ----------
    X: ndarray
        Frequency 2D meshgrid array.
    Y: ndarray
        Frequency 2D meshgrid array.
    Z: complex ndarray
        Complex 2D field amplitudes.
    deg:integer,optional
        Degree of the polynomial to fit. 
    origin: tuple (float,float),optional
        Manually set origin of the distribution.
    method:{'full','sep'}
        Use 'full' to fit to the entire 2D polynomial. Use 'sep' to fit to
        a separable polynomial which sets all cross terms to 0.
    tol: float, optional
        Cutoff intensity for phase fit in percentage of field intensity.
    plot:bool,optional 
        Plot fit results.
    fits:bool,optional
        Apply and print fits.
        

    Returns
    ----------
    out:tuple (ndarray,ndarray)
        Optimal values for the parameters so that the sum of the squared
        residuals of ``f(xdata, *popt) - ydata`` is minimized
    '''

    #phase=unwrap_phase(np.angle(Z))

    phase=unwrap_phase(np.angle(Z))
    phase[np.where(np.abs(Z)**2/np.max(np.abs(Z)**2)<=tol)]=0
    #phase=unwrap_phase(phase)

    #phase=Z

    #Average non-zero elements
    xm=X[0,:]
    ym=Y[:,0]

    #zmx=unwrap(angle.sum(0)/((angle!=0).sum(0)+0.001))
    #zmx=unwrap(angle.sum(0))#/((angle!=0).sum(0)+0.001)
    #zm=(angle.sum(0))#/((angle!=0).sum(0)+0.001)
    #zm=np.unwrap(angle).sum(0)/((angle!=0).sum(0)+0.001)
    #zmy=np.unwrap(angle.sum(1))#/((angle!=0).sum(0)+0.001)

    #zmx=np.unwrap(np.unwrap(angle,axis=0).sum(0))#/((angle!=0).sum(0)+0.001)
    #zmy=np.unwrap(np.unwrap(angle,axis=1).sum(1))#/((angle!=0).sum(0)+0.001)

    zmx=np.trapz(phase,x=ym,axis=0)/(ym.max()-ym.min())
    zmy=np.trapz(phase,x=xm,axis=1)/(xm.max()-xm.min())

    zmx=np.trapz(phase,x=ym,axis=0)/(np.diff(ym)[0]*(phase!=0).sum(0))
    zmy=np.trapz(phase,x=xm,axis=1)/(np.diff(xm)[0]*(phase!=0).sum(1))

    zmx[np.where(np.isnan(zmx))]=0
    zmy[np.where(np.isnan(zmy))]=0
    
    roix=np.where(zmx)
    roiy=np.where(zmy)
    #zmx=phase.sum(0)
    #zmy=phase.sum(1)

    if len(index)==4:
        xmin,xmax,ymin,ymax=index
    else:
        xmin,xmax,ymin,ymax=0,None,0,None

    if fits:
        px,_=curve_fit(poly3,xm[roix][xmin:xmax],zmx[roix][xmin:xmax],p0[0])

        py,_=curve_fit(poly3,ym[roiy][ymin:ymax],zmy[roiy][ymin:ymax],p0[1])

    if plots:

        fig,ax=plt.subplots(1,3,figsize=(12,4))
        S=ax[0].pcolormesh(X,Y,phase)
        ax[1].plot(xm,zmx,'.C1')

        ax[2].plot(ym,zmy,'.C1')

        if fits:
            ax[1].plot(xm[roix][xmin:xmax],poly3(xm[roix][xmin:xmax],*px),'C1')
            ax[2].plot(ym[roiy][ymin:ymax],poly3(ym[roiy][ymin:ymax],*py),'C1')

        ax[0].set_xlabel(r'$\omega_i$',fontsize=18)
        ax[0].set_ylabel(r'$\omega_s$',fontsize=18)

        ax[1].set_xlabel(r'$\omega_i$',fontsize=18)
        ax[1].set_ylabel(r'$\phi_i$',fontsize=18)

        ax[2].set_xlabel(r'$\omega_s$',fontsize=18)
        ax[2].set_ylabel(r'$\phi_s$',fontsize=18)

        fig.colorbar(S,ax=ax[0]) 

        plt.tight_layout()
        plt.show()

        if fits:
            print(px)
            print(py) 

    if fits:    
        return(px,py) 
    
def polyfit2phase(Wi,Ws,Fk,
                  method='sep',deg=3,tol=0.1,origin=0,plot=True,unwrap=True):
    '''
    Fit the spectral phase of a 2D complex field Fk to a polynomial of the
    specified degree.

    Parameters
    ----------
    Wi: ndarray
        Frequency 2D meshgrid array.
    Ws: ndarray
        Frequency 2D meshgrid array.
    Fk: complex ndarray
        Complex 2D field amplitudes.
    deg:integer,optional
        Degree of the polynomial to fit. 
    origin: tuple (float,float),optional
        Manually set origin of the distribution.
    method:{'full','sep'}
        Use 'full' to fit to the entire 2D polynomial. Use 'sep' to fit to
        a separable polynomial which sets all cross terms to 0.
    tol: float, optional
        Cutoff intensity for phase fit in percentage of field intensity.
    plot:bool,optional 
        Plot fit results.
    unwrap: bool, optional
        Unwrap 2D phase using skimage.restoration.unwrap_phase function.

    Returns
    ----------
    coef : ndarray, shape (deg[0] + 1, deg[1] +1) 
        Polynomial fit coefficients ordered from low to high.
    '''
    
    #Fk=Fk/np.max(np.abs(Fk)**2)
    nroi=np.where(np.abs(Fk)**2/np.max(np.abs(Fk)**2)<tol)
    roi=np.where(np.abs(Fk)**2/np.max(np.abs(Fk)**2)>=tol)

    if unwrap:
        phase=unwrap_phase(np.angle(Fk))
    else:
        phase=(np.angle(Fk))
    
    phase0=phase.copy()
    phase0[nroi]=0
    
    if origin:
        wi0,ws0=origin
    else:
        Ik=np.abs(Fk)**2/np.max(np.abs(Fk)**2)
        wi0,ws0=fit_gaussian((Wi,Ws,Ik+0.001*np.max(Ik))).x[1:3]

    if method=='sep':
        coeff,cost,_,_=polyfit2sep(Wi[roi]-wi0,Ws[roi]-ws0,phase[roi],deg=deg)
    elif method=='full':
        coeff,cost,_,_=polyfit2d(Wi[roi]-wi0,Ws[roi]-ws0,phase[roi],deg=[deg,deg])
    fc=poly.polyval2d(Wi-wi0,Ws-ws0,coeff)
    fc[nroi]=0

    if plot:
        fig,ax=plt.subplots(1,2,figsize=(9,4))
        S1=ax[0].pcolormesh(Wi-wi0,Ws-ws0,phase0)
        S2=ax[1].pcolormesh(Wi-wi0,Ws-ws0,fc)
        ax[0].set_title('Reconstructed')
        ax[1].set_title('Fit')
        fig.colorbar(S1,ax=ax[0])
        fig.colorbar(S2,ax=ax[1])
        plt.tight_layout()
        plt.show()

        print('Cost\n',cost[0],'\nCentre\n',wi0,ws0,'\nCoeff\n',coeff)

    return coeff

def weighted_polyfit2phase(Wi,Ws,Fk,
                           p0=[],plots=False):

    '''
    !!!
    I don't know that this is used anywhere. Should we delete this
    function.
    !!!

    Apply weighted fit to the spectral phase of a 2D complex field Fk to a polynomial of the
    specified degree.

    Parameters
    ----------
    Wi: ndarray
        Frequency 2D meshgrid array.
    Ws: ndarray
        Frequency 2D meshgrid array.
    Fk: complex ndarray
        Complex 2D field amplitudes.

    Returns
    ----------

    '''
    weights=np.abs(Fk)**2
    phase=unwrap_phase(np.angle(Fk))
    weighted_phase=weights*phase

    wi0,ws0=fit_gaussian((Wi,Ws,np.abs(Fk)**2)).x[1:3]

    if len(p0)==0:
        p0=np.random.rand(1,7)[0]

    res=least_squares(wpoly2d_cost,p0,args=((Wi-wi0).reshape(-1),(Ws-ws0).reshape(-1),(weights).reshape(-1),weighted_phase.reshape(-1)),method='lm')

    if plots:
        fig,ax=plt.subplots(1,2,figsize=(9,4))
        ax=ax.reshape(-1)
        S1=ax[0].pcolormesh(Wi-wi0,Ws-ws0,weighted_phase)
        S2=ax[1].pcolormesh(Wi-wi0,Ws-ws0,wpoly2d([Wi-wi0,Ws-ws0,weights],*res.x))
        [ax[i].set_title(['Measured','Least_squares'][i]) for i in range(2)]
        [fig.colorbar([S1,S2][i],ax=ax[i]) for i in range(2)]
        plt.tight_layout()
        plt.show()

        print('\ncost\n',res.cost,'\nleast_squares\n',res.x)
    
    return res
