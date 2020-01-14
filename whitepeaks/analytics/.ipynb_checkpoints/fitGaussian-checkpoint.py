'''
###############################################################################
FIT GAUSSIAN MODULE
###############################################################################

This module contains the following functions:

Fit functions
-------------
> fit_gaussian(data)
> fit_errors(function,p0,data,sample_size)

Get functions
-------------
> get_gaussian_moments(data)
> get_correlation(X,Y,Z)
> gaussian_parameters(X,Y,Z)
> get_slice(x,z,y,yslice
> get_heralded_moments(X,Z,Y,yslice,value)
> get_mean_heralded_moments(X,Z,Y)
> print_parameters(X,Y,Z,gateX,gateY

Deconvolve functions
--------------------
> deconvolve_gaussian2d(xsigma,ysigma,rho,rf_x,rf_y)
> deconvolve_heralded_width(hwidthx,hwidthy,rho,responsefx,responsefy)

'''

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares, curve_fit

from .functions import *
from .stats import *

'''
-------------------------------------------------------------------------------
Fit functions
-------------------------------------------------------------------------------
'''
def fit_gaussian(data):
    '''
    Fit data to gaussian function. 
    
    Parameters 
    ----------
    data: tuple (ndarray,ndarray) or tuple(ndarray,ndarray,ndarray)
        x,y coordinates of the gaussian distribution or (X,Y,Z) meshgrid
        of 2D distribution.

    Returns
    ----------
    out: dict 
        Same output as scipy.optimize.least_squares.
    '''

    n=np.shape(data)[0]

    if n==2:
        x,y=data
        x0,xsigma=get_gaussian_moments((x,y))
        p0=[y.max(),x0,xsigma,y.min()]
        res=least_squares(gauss_cost,p0,args=(x,y),bounds=([0,-np.inf,0,0],np.inf))

        return res

    elif n==3:
        X,Y,Z=data
        if len(np.shape(X))==2:
            x0,xsigma=get_gaussian_moments(get_marginal(X,Z))
            y0,ysigma=get_gaussian_moments(get_marginal(Y.T,Z.T))
        elif len(np.shape(X))==1:
            x0,xsigma=get_moments(X,Z)
            y0,ysigma=get_moments(Y,Z)

        rho=1/np.sum(Z)*np.sum(Z*(X-x0)*(Y-y0))/(xsigma*ysigma)
        p0=[Z.max(),x0,y0,xsigma,ysigma,rho,Z.min()]

        res=least_squares(gauss2d_cost,p0,args=(X.reshape(-1),Y.reshape(-1),Z.reshape(-1)))
        #if len(np.shape(X))==2:
        #    res=least_squares(gauss2d_cost,p0,args=(X.reshape(-1),Y.reshape(-1),Z.reshape(-1)))

        #elif len(np.shape(X))==1:
        #    res=least_squares(gauss2d_cost,p0,args=(X,Y,Z))

        return res

def fit_errors(function,p0,data,sample_size):
    n=np.shape(data)[0]
    poisson_res=np.zeros((sample_size,len(p0)))

    if n==2:
        x,y=data
        for i in range(sample_size):
            resp=least_squares(function,p0,args=(x,np.random.poisson(y)),
                               bounds=([0,-np.inf,0,0],np.inf))
            poisson_res[i,]=resp.x
    if n==3:
        x,y,z=data
        for i in range(sample_size):
            resp=least_squares(function,p0,args=(x,y,np.random.poisson(z)),
                               bounds=([0,-np.inf,-np.inf,0,0,-1,0],[np.inf,np.inf,np.inf,np.inf,np.inf,1,np.inf]))
            #if ~(np.isnan(resp.x).any()):
            poisson_res[i,]=resp.x

    res_std=np.std(poisson_res,0) 
    res_mean=np.mean(poisson_res,0)
    
    return res_mean,res_std

'''
-------------------------------------------------------------------------------
Get functions
-------------------------------------------------------------------------------
'''
def get_gaussian_moments(data,
                         get_errors=False,sample_size=100):
    '''
    Calculate the moments of a gaussian distribution. 
    
    Parameters 
    ----------
    data: tuple (ndarray,ndarray)
        x,y coordinates of the gaussian distribution.
    get_errors:bool,optional 
        Calculate errors on moments.
    sample_size: int, optional
        Sample size for error calculation.

    
    Returns
    ----------
    out: tuple(float,float)
        (x0,sigma) of gaussian distribution.
    '''

    x,y=data
    xmean=np.sum(y*x)/np.sum(y)
    xsigma=np.sqrt(1.0/np.sum(y)*np.sum(y*(x-xmean)**2))

    p0=[y.max(),xmean,xsigma,y.min()]
    mres=least_squares(gauss_cost,p0,args=(x,y),bounds=([0,-np.inf,0,0],np.inf))

    if get_errors:
        poisson_res=np.zeros((sample_size,len(p0)))

        for i in range(sample_size):

            resp=least_squares(gauss_cost,p0,args=(x,np.random.poisson(y)),bounds=([0,-np.inf,0,0],np.inf))
            poisson_res[i,]=resp.x

        res_std=np.std(poisson_res,0) 
        res_mean=np.mean(poisson_res,0)

        return mres.x[1],mres.x[2],res_std[1],res_std[2]

    else:

        return mres.x[1], mres.x[2]

def get_correlation(X,Y,Z,
                    get_errors=False,sample_size=100,output_all=False):

    x0,xsigma=get_gaussian_moments(get_marginal(X,Z))
    y0,ysigma=get_gaussian_moments(get_marginal(Y.T,Z.T))

    rho=correlation(X,Y,Z)

    p0=[Z.max(),x0,y0,rho,Z.min()]

    res=least_squares(
        lambda p,x,y,z:(z-gauss2d([x,y],p[0],p[1],p[2],xsigma,ysigma,p[3],p[4]))/
                       (np.sqrt(gauss2d([x,y],p[0],p[1],p[2],xsigma,ysigma,p[3],p[4]))+0.001),
                       p0,args=(X.reshape(-1),Y.reshape(-1),Z.reshape(-1)),
                       bounds=([0,-np.inf,-np.inf,-1,0],[np.inf,np.inf,np.inf,1,np.inf]))

    if get_errors:
        poisson_res=np.zeros((sample_size,len(p0)))
        for i in range(sample_size):

            resp=least_squares(
            lambda p,x,y,z:(z-gauss2d([x,y],p[0],p[1],p[2],xsigma,ysigma,p[3],p[4]))/
                           (np.sqrt(gauss2d([x,y],p[0],p[1],p[2],xsigma,ysigma,p[3],p[4]))+0.001),
                           p0,args=(X.reshape(-1),Y.reshape(-1),np.random.poisson(Z.reshape(-1))),
                bounds=([0,-np.inf,-np.inf,-1,0],[np.inf,np.inf,np.inf,1,np.inf]))

            poisson_res[i,]=resp.x

        res_std=np.std(poisson_res,0) 
        res_mean=np.mean(poisson_res,0)

        if output_all: return res.x,res_std
        else: return res.x[3],res_std[3]
    else:
        if output_all: 
            return res.x[0],res.x[1],res.x[2],xsigma,ysigma,res.x[3],res.x[4]
        else: return res.x[3]

def gaussian_parameters(X,Y,Z):
    """
    Get the 2d-gaussian parameters without errors
    """

    #Marginals
    xm,zmx=get_marginal(X,Z)
    ym,zmy=get_marginal(Y.T,Z.T)

    #Moments of the distribution
    xm0,xmsigma=get_moments(xm,zmx)
    ym0,ymsigma=get_moments(ym,zmy)
    rhom=1.0/np.sum(Z)*np.sum(Z*(X-xm0)*(Y-ym0))/(xmsigma*ymsigma)

    #Fit marginal parameters
    x0,xsigma=get_gaussian_moments((xm,zmx))
    y0,ysigma=get_gaussian_moments((ym,zmy))
    rho=get_correlation(X,Y,Z)

    return x0,y0,xsigma,ysigma,rho

def get_slice(x,z,y,yslice,
              value=0.02):
    yroi=np.where(np.abs(y-yslice)<=value)
    return x[yroi],z[yroi]

def get_heralded_moments(X,Z,Y,yslice,value,
                         get_errors=True,sample_size=100):
    xs,zs=get_slice(X,Z,Y,yslice,value)
    fit=fit_gaussian((xs,zs))

    if get_errors:
        errors=fit_errors(gauss_cost,fit.x,(xs,zs),sample_size)
        return fit.x[1],fit.x[2],errors[1][1],errors[1][2]
    else:
        return fit.x[1],fit.x[2]

def get_mean_heralded_moments(X,Z,Y):
    """
    Get the average heralded width of X,Z for different fixed values of Y.
    """
    dy=np.abs(np.diff(Y.T).mean()) #Find the spacing between slices
    y0,ysigma=get_gaussian_moments(get_marginal(Y.T,Z.T)) #Get moments
    yslice=np.arange(y0-ysigma,y0+ysigma+dy,dy) #Define range in terms of moments, take slices to 1 std
    hwidth=np.zeros(yslice.size)
    for i in np.arange(yslice.size):
        hwidth[i]=get_heralded_moments(X,Z,Y,yslice[i],dy/2)[1]
        #print yslice[i],hwidth[i]
    return hwidth.mean(),hwidth.std()

def print_parameters(X,Y,Z,gateX,gateY,
                     sample_size=100,deconvolve=True):
    """
    Get the 2d-gaussian parameters with errors
    """
    dsi,ddsi=gateX
    dss,ddss=gateY

    #Marginals
    xm,zmx=get_marginal(X,Z)
    ym,zmy=get_marginal(Y.T,Z.T)

    #Moments of the distribution
    xm0,xmsigma=get_moments(xm,zmx)
    ym0,ymsigma=get_moments(ym,zmy)
    rhom=1.0/np.sum(Z)*np.sum(Z*(X-xm0)*(Y-ym0))/(xmsigma*ymsigma)

    #Fit marginal parameters
    x0,xsigma,dx0,dxsigma=get_gaussian_moments((xm,zmx),get_errors=True,sample_size=sample_size)
    y0,ysigma,dy0,dysigma=get_gaussian_moments((ym,zmy),get_errors=True,sample_size=sample_size)
    rho,drho=get_correlation(X,Y,Z,get_errors=True,sample_size=sample_size)

    #Fit heralded parameters
    #diffy=np.abs(np.diff(Y.T)[0][0])
    #diffx=np.abs(np.diff(X)[0][0])
    #xh0,xhsigma,dxh0,dxhsigma=get_heralded_moments(X,Z,Y,y0,diffy/2,get_errors=True,sample_size=sample_size)
    #yh0,yhsigma,dyh0,dyhsigma=get_heralded_moments(Y,Z,X,x0,diffx/2,get_errors=True,sample_size=sample_size)

    #Mean heralded moments

    xhsigma,dxhsigma=get_mean_heralded_moments(X,Z,Y)
    yhsigma,dyhsigma=get_mean_heralded_moments(Y.T,Z.T,X.T)

    if deconvolve:
        #Deconvolved parameters
        (xsigma_prime,ysigma_prime,rho_prime,
         dxsigma_prime,dysigma_prime,drho_prime)=deconvolve_gaussian2d(xsigma,ysigma,rho,dsi,dss,
                                                         errors=[dxsigma,dysigma,drho,ddsi,ddss])
        #Deconvolve heralded widths
        xhsigma_prime,dxhsigma_prime=deconvolve_heralded_width(xhsigma,yhsigma,rho,dsi,dss,
                                                               errors=[dxhsigma,dyhsigma,drho,ddsi,ddss])

        yhsigma_prime,dyhsigma_prime=deconvolve_heralded_width(yhsigma,xhsigma,rho,dss,dsi,
                                                               errors=[dyhsigma,dxhsigma,drho,ddss,ddsi])
    
    #print('\nDistribution moments')
    #print('X (rms): %f %f' %(xm0,xmsigma))
    #print('Y (rms): %f %f' %(ym0,ymsigma))
    #print('Correlation:%f' %(rhom))

    print("\nRaw fit parameters")
    print('Y0: %f %f'%(y0,dy0))
    print('X0: %f %f'%(x0,dx0))
    print('Y marginal width (rms): %f %f'%(ysigma,dysigma))
    print('Y heralded width (rms): %f %f'%(yhsigma,dyhsigma))
    print('X marginal width (rms): %f %f'%(xsigma,dxsigma))
    print('X heralded width (rms): %f %f'%(xhsigma,dxhsigma))
    print('Correlation:%f %f'%(rho,drho))

    if deconvolve:
        print('\nDeconvolved parameters')
        print('Y marginal (rms):%f %f nm'%(ysigma_prime,dysigma_prime))
        print('Y heralded width (rms): %f %f'%(yhsigma_prime,dyhsigma_prime))
        print('X marginal (rms):%f %f nm'%(xsigma_prime,dxsigma_prime))
        print('X heralded width (rms): %f %f'%(xhsigma_prime,dxhsigma_prime))
        print('Correlation: %f %f'%(rho_prime,drho_prime) )

'''
-------------------------------------------------------------------------------
Deconvolve functions
-------------------------------------------------------------------------------
'''
def deconvolve_gaussian2d(xsigma,ysigma,rho,rf_x,rf_y,
                          errors=[]):

    dx,dy=rf_x,rf_y #Response functions

    xsigma_prime=(np.sqrt(xsigma**2-dx**2))

    ysigma_prime=(np.sqrt(ysigma**2-dy**2))

    rho_prime=(rho*xsigma*ysigma/(
                np.sqrt(ysigma**2-dy**2)*np.sqrt(xsigma**2-dx**2)))


    if errors:
        dxsigma,dysigma,drho,ddx,ddy=errors

        dxsigma_prime=np.sqrt((dxsigma*xsigma/xsigma_prime)**2+(ddx*dx/xsigma_prime)**2)

        dysigma_prime=np.sqrt((dysigma*ysigma/ysigma_prime)**2+(ddy*dy/ysigma_prime)**2)

        drho_prime=np.sqrt(rho_prime**2*(
                    + (dx**2/xsigma/(xsigma**2-dx**2)*dxsigma)**2
                    + (dy**2/ysigma/(ysigma**2-dy**2)*dysigma)**2
                    + (drho/rho)**2
                    + (dx/(xsigma**2-dx**2)*ddx)**2
                    + (dy/(ysigma**2-dy**2)*ddy)**2))

        return xsigma_prime,ysigma_prime,rho_prime,dxsigma_prime,dysigma_prime,drho_prime

    else:

        return xsigma_prime,ysigma_prime,rho_prime

def deconvolve_heralded_width(hwidthx,hwidthy,rho,responsefx,responsefy,
                              errors=[]):
    hwx,hwy,rho,rfx,rfy=hwidthx,hwidthy,rho,responsefx,responsefy

    a=np.sqrt((hwy**2-rfy**2)/(hwy**2-(1-rho**2)*rfy**2))
    #print a
    hwx_prime=np.sqrt(a**2*hwx**2-rfx**2)
    

    if errors:
        dhwx,dhwy,drho,drfx,drfy=errors

        da=np.sqrt(rfy**2*rho**2*(rho**2*hwy**2*(dhwy**2*rfy**2+drfy**2*hwy**2)+drho**2*rfy**2*(hwy**2-rfy**2)**2)
             /((hwy**2-rfy**2)*(hwy**2-(1-rho**2)*rfy**2)**3))

        dhwx_prime=np.sqrt((drfx**2*rfx**2+a**2*hwx**2*(a**2*dhwx**2+da**2*hwx**2))
                      /(a**2*hwx**2-rfx**2))
        #print a,da

        return hwx_prime,dhwx_prime
    else:
        return hwx_prime
    