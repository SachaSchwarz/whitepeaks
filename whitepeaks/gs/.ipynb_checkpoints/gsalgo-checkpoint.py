'''
###############################################################################
                        #GERCHBERG SAXTON ALGORITHM FUNCTIONS 
###############################################################################

A list of functions for applying the Gerchberg-Saxton algorithm on 2D
arrays of conjugate data.

Routines in this module:

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares 
from scipy.stats import entropy 
from skimage.restoration import unwrap_phase

from .io import *
from ..fit import *
from .binner import *

from matplotlib.colors import LinearSegmentedColormap

cmap_darkcyan=LinearSegmentedColormap.from_list('my_colormap',['black','darkcyan','white'])
cmap_dodgerblue=LinearSegmentedColormap.from_list('my_colormap',['black','dodgerblue','white'])

colormaps=['gist_heat',cmap_darkcyan,cmap_darkcyan,cmap_dodgerblue]
colormaps=['jet','jet','jet','jet']
colormaps=['viridis','viridis','viridis','viridis']

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

def GS_modified(Iww,Itt,Itw,Iwt,initial_guess='random',max_iter=5000,tol=1e-5,
                 error_model='NRMS',return_errors=False,method='modified',
                 target_state=np.array([]),x=np.array([]),y=np.array([])):
    '''
    Gerchberg-Saxton algorithm for finding 2D phase from the modulus of 
    the joint spectral, joint temporal, and time-frequency data.

    Parameters
    ----------
    Iww: ndarray 
        2D array of Frequency-Frequency count data.     
    Itt: ndarray 
        2D array of Time-Time data.     
    Itw: ndarray 
        2D array of Time-Frequency data.     
    Iww: ndarray 
        2D array of Frequency-Time data.     
    max_iter: int,optional
        Maximum number of loops in algorithm
    initial_guess:{'random','flat',2darray}
    method: {'modified','gs'}, optional
        To run the regular gs algorithm which only uses
        frequency-frequency and time-time data use the 'gs' option. To use
        all time-frequency data use the 'modified' option.
    error_model: {'FROG', 'NRMS', 'KL'} str, optional
        Option to use FROG, Normalized root-mean square Kullback-Leibler
        divergence to calculate error between reconstructed and measured
        state.
    tol:float,optional 
        stop iterations when error<tol.
    target_state: tuple (Wi,Ws,F), optional 
        Target state for fidelity calculation.

    Returns
    ---------- 
    out: tuple with following variables
        i:  int
            Number of iterations
        fww:ndarray
            2D array of complex frequency-frequency amplitudes.
        ftw:ndarray
            2D array of complex time-frequency amplitudes.
        fwt:ndarray
            2D array of complex frequency-time amplitudes.
        ftt:ndarray
            2D array of complex time-time amplitudes.
        Errors:1D array,optional 
            Error at each iteration. Only included in output if
            return_errors is True.
        fid: 1D array,optional
            Fidelity at each iteration. Only included in output if target
            state is specified.  
    '''

    Itt=np.double(Itt)
    Iww=np.double(Iww)
    Itw=np.double(Itw)
    Iwt=np.double(Iwt)
    
    #Check that normalization of modulus data satifies Parseval's theorem
    Itt*=np.sum(Iww)/np.sum(Itt)
    Itw*=np.sum(Iww)/np.sum(Itw)
    Iwt*=np.sum(Iww)/np.sum(Iwt)

    #Initialize function
    if isinstance(initial_guess,str):
        if initial_guess=='random':
            Fww=np.sqrt(Iww)*random_phase(Iww)
        elif initial_guess=='flat': 
            Fww=np.sqrt(Iww)
    else: 
        Fww=initial_guess

    #Initialize iteration
    EF=100
    Ef=100
    
    if return_errors:
        Errors=np.zeros((max_iter,2))
        
    if target_state.size!=0:
        fid=np.zeros((max_iter,2))
    
    i=0
    while (EF>tol or Ef>tol):
        if i>=max_iter:
            i-=1
            #print 'Maximum iterations reached'
            break
        
        #Calculate joint temporal amplitude
        ftw1=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Fww),norm='ortho',axes=[1]))
        
        #Replace modulous with measured spectral values
        if method=='gs':
            ftw2=ftw1
        elif method=='modified' and (i<500 or i%250<10) and i<3000:
            ftw2=ftw1/(np.abs(ftw1)+1e-4*np.max(np.abs(ftw1)))*np.sqrt(Itw)
        else: ftw2=ftw1

        #Calculate joint spectral amplitude
        ftt1=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ftw2),norm='ortho',axes=[0]))
        
        #Replace modulous with measured spectral values
        ftt2=ftt1/(np.abs(ftt1)+1e-4*np.max(np.abs(ftt1)))*np.sqrt(Itt)

        #Calculate joint spectral amplitude
        fwt1=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ftt2),norm='ortho',axes=[1]))

        #Replace modulous with measured spectral values
        if method=='gs':
            fwt2=fwt1
        elif method=='modified' and (i<500 or i%250<10) and i<3000:
            fwt2=fwt1/(np.abs(fwt1)+1e-4*np.max(np.abs(fwt1)))*np.sqrt(Iwt)
        else: fwt2=fwt1

        #Calculate joint spectral amplitude
        fww1=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(fwt2),norm='ortho',axes=[0]))

        #Replace modulous with measured spectral values
        fww2=fww1/(np.abs(fww1)+1e-4*np.max(np.abs(fww1)))*np.sqrt(Iww)

        #Calculate error
        if error_model=='NRMS':
            #print np.isnan(Iww).any(),np.isnan(Fk1).any()
            EF,muF=NRMS(np.sqrt(Iww),np.abs(fww1))
            Ef,muf=NRMS(np.sqrt(Itt),np.abs(ftt1))
        elif error_model=='FROG':
            #USE FROG ERROR
            EF,muF=FROG_error(Iww,np.abs(fww1)**2)      
            Ef,muf=FROG_error(Itt,np.abs(ftt1)**2)
        elif error_model=='KL':
            #USE KL divergence or relative entropy
            EF=np.sum(entropy(Iww,np.abs(fww1)**2))
            Ef=np.sum(entropy(Itt,np.abs(ftt1)**2))
        
        if return_errors:
            Errors[i,:]=np.array([EF,Ef])
        
        if target_state.size!=0:
            fid[i,0]=fidelity(target_state,fww1,(x,y))
            fid[i,1]=fidelity(np.conjugate(target_state),ftt1,(x,y))
            
        Fww=fww2
        i+=1
    
    if return_errors and target_state.size!=0:
        return i+1,fww1,ftw1,fwt1,ftt1,Errors,fid
    elif return_errors:
        return i+1,fww1,ftw1,fwt1,ftt1,Errors
    elif target_state.size!=0:
        return i+1,fww1,ftw1,fwt1,ftt1,fid
    else:
        return i+1,fww1,ftw1,fwt1,ftt1


def run_gs_algorithm(gs_input,max_iter,method='modified',error_model='FROG',initial_guess='random',tol=1E-5,
                    target_state=[]):
    '''
    Run the Modified GS algorithm and print the output.
    For calculating fidelities, target_state should be of the form
    [x,y,F].

    Parameters
    ----------
    gs_input:tuple 
        (Wi,Ws,Iww,Ti1,Ws1,Itw,Wi2,Ts2,Iwt,Ti,Ts,Itt)
        or
        (Wi,Ws,Ti,Ts,Iww,Itw,Iwt,Itt)
    max_iter: int,optional
        Maximum number of loops in algorithm
    initial_guess:{'random','flat',2darray}
    method: {'modified','gs'}, optional
        To run the regular gs algorithm which only uses
        frequency-frequency and time-time data use the 'gs' option. To use
        all time-frequency data use the 'modified' option.
    error_model: {'FROG', 'NRMS', 'KL'} str, optional
        Option to use FROG, Normalized root-mean square Kullback-Leibler
        divergence to calculate error between reconstructed and measured
        state.
    tol:float,optional 
        stop iterations when error<tol.
    target_state: tuple (Wi,Ws,F), optional 
        Target state for fidelity calculation.

    Returns
    ---------- 
    output: tuple with following variables
        i:  int
            Number of iterations
        fww:ndarray
            2D array of complex frequency-frequency amplitudes.
        ftw:ndarray
            2D array of complex time-frequency amplitudes.
        fwt:ndarray
            2D array of complex frequency-time amplitudes.
        ftt:ndarray
            2D array of complex time-time amplitudes.
        Errors:1D array,optional 
            Error at each iteration. Only included in output if
            return_errors is True.
        fid: 1D array,optional
            Fidelity at each iteration. Only included in output if target
            state is specified.  
    '''

    if len(gs_input)==8:
        Wi,Ws,Ti,Ts,Iww,Itw,Iwt,Itt=gs_input
        Ti1,Ws1=Ti,Ws
        Wi2,Ts2=Wi,Ts
    elif len(gs_input)==12:
        Wi,Ws,Iww,Ti1,Ws1,Itw,Wi2,Ts2,Iwt,Ti,Ts,Itt=gs_input

    if len(target_state)==3:
        
        #Calculate fidelity with target state for simulated data.

        x,y,F=target_state
        output=GS_modified(Iww,Itt,Itw,Iwt,
                       initial_guess=initial_guess,method=method,tol=tol,
                       max_iter=max_iter,error_model=error_model,return_errors=True,
                       target_state=F,x=x,y=y)

        i,Fk,fktw,fkwt,fk,error,fid=output

    else:

        output=GS_modified(Iww,Itt,Itw,Iwt,
                       initial_guess=initial_guess,method=method,tol=tol,
                       max_iter=max_iter,error_model=error_model,return_errors=True)

        i,Fk,fktw,fkwt,fk,error=output

    print('Iterations',i)
    print('Grid_size',Fk.shape)

    xlabels=[r'Idler Frequency (fs$^{-1}$)','Idler delay (fs) ',r'Idler Frequency (fs$^{-1})$','Idler delay (fs) ']
    ylabels=[r'Signal Frequency (fs$^{-1}$)',r'Signal Frequency (fs$^{-1}$)','Signal delay (fs)','Signal delay (fs)']
    xlabels=[r'$\omega_i$ (fs$^{-1}$)',r'$t_i$ (fs) ',r'$\omega_i$ (fs$^{-1})$',r'$t_i$ (fs) ']
    ylabels=[r'$\omega_s$ (fs$^{-1}$)',r'$\omega_s$ (fs$^{-1})$',r'$t_s$ (fs) ',r'$t_s$ (fs) ']
    
    for k in range(4): 
        X=[Wi,Ti1,Wi2,Ti][k]
        Y=[Ws,Ws1,Ts2,Ts][k]
        I=[Iww,Itw,Iwt,Itt][k]
        F=[Fk,fktw,fkwt,fk][k]

        Fk_angle=np.angle(F)
        Fk_angle[np.where((np.abs(F)**2/np.max(np.abs(F)**2))<0.05)]=0

        fig,ax=plt.subplots(1,3,figsize=(12,3))
        ax=ax.reshape(-1)
        S0=ax[0].pcolormesh(X,Y,I,cmap=colormaps[k])
        S1=ax[1].pcolormesh(X,Y,np.abs(F)**2,cmap=colormaps[k])
        S2=ax[2].pcolormesh(X,Y,Fk_angle,cmap=colormaps[k])

        [fig.colorbar([S0,S1,S2][j],ax=ax[j]) for j in range(3)]  
        if k==0 and 1:
            ax[0].set_title('Measured Intensity')
            ax[1].set_title('Reconstructed Intensity')
            ax[2].set_title('Reconstructed Phase')

        [ax[i].set_xlabel(xlabels[k],fontsize=18) for i in range(3)] 
        [ax[i].set_ylabel(ylabels[k],fontsize=18) for i in range(3)] 
        plt.tight_layout()
        plt.show()
    
        
    if len(target_state)==3:
        #Make two plots
        fig,ax=plt.subplots(2,1,figsize=(10,6))
        ax[0].semilogy(error[:,0],label=r'$F_k(\omega_i,\omega_s)$')
        ax[0].semilogy(error[:,1],label=r'$f_k(t_i,t_s)$')
        ax[1].plot(fid[:,0],label=r'$F_k(\omega_i,\omega_s)$')
        ax[1].plot(fid[:,1],label=r'$F_k(\omega_i,\omega_s)^*$')
        ax[0].set_title(error_model)
        ax[0].legend()
        ax[1].set_ylim(0,1)
        ax[1].set_title('Fidelity')        
        ax[1].legend()
        plt.tight_layout()
        plt.show()
    else:
        #Make one plot
        fig,ax=plt.subplots(1,1,figsize=(10,6))
        ax.semilogy(error[:,0],label=r'$F_k(\omega_i,\omega_s)$')
        ax.semilogy(error[:,1],label=r'$f_k(t_i,t_s)$')
        ax.set_title(error_model)
        ax.legend()
        plt.tight_layout()
        plt.show()

    print('FROG',FROG_error(Iww,np.abs(Fk)**2)[0],FROG_error(Itt,np.abs(fk)**2)[0])
    print('NRMS',NRMS(np.sqrt(Iww),np.abs(Fk))[0],NRMS(np.sqrt(Itt),np.abs(fk))[0])
    print('KL',np.sum(entropy(Iww,np.abs(Fk)**2)),np.sum(entropy(Itt,np.abs(fk)**2)))

    if len(target_state)==3:
        #If fidelity was calculated
        print('Fidelity',fid[-1,:])

    return output

def fit_marginal_phase(X,Y,Z,index=[],tol=0.1,p0=[None,None],plots=True,fits=True):

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

    #zmx=phase.sum(0)
    #zmy=phase.sum(1)

    if len(index)==4:
        xmin,xmax,ymin,ymax=index
    else:
        xmin,xmax,ymin,ymax=0,-1,0,-1

    if fits:
        px,_=curve_fit(poly3,xm[xmin:xmax],zmx[xmin:xmax],p0[0])

        py,_=curve_fit(poly3,ym[ymin:ymax],zmy[ymin:ymax],p0[1])

    if plots:

        fig,ax=plt.subplots(1,3,figsize=(12,4))
        S=ax[0].pcolormesh(X,Y,phase)
        ax[1].plot(xm,zmx,'.C1')

        ax[2].plot(ym,zmy,'.C1')

        if fits:
            ax[1].plot(xm[xmin:xmax],poly3(xm[xmin:xmax],*px),'C1')
            ax[2].plot(ym[ymin:ymax],poly3(ym[ymin:ymax],*py),'C1')

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
        return(px,py) 
    
def polyfit2phase(Wi,Ws,Fk,tol=0.1,plot=True,unwrap=True,origin=0,deg=3):

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
        wi0,ws0=fit_gaussian((Wi,Ws,np.abs(Fk)**2)).x[1:3]

    coeff,cost,_,_=polyfit2sep(Wi[roi]-wi0,Ws[roi]-ws0,phase[roi],deg=deg)
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

def weighted_polyfit2phase(Wi,Ws,Fk,p0=[],plots=False):
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

#Code snippets
#   #Constrain the time-frequency plots
#   #Hybrid input output
#   if 0:
#       if 0 and i>50 and (i-50)%50<40:
#           ftw2[roi1]=np.abs(ftw2[roi1])-beta*np.abs(ftw1[roi1])
#       elif i<100:
#           ftw2=ftw1
#           ftw2[roi1]=0
#       else: ftw2=ftw1
#
#    #Constrain the frequency-time plot
#    if 0:
#        if 0 and i>50 and (i-50)%50<40:
#            fwt2[roi2]=np.abs(fwt2[roi2])-beta*np.abs(fwt1[roi2])
#        elif i<100:
#            fwt2=fwt1
#            fwt2[roi2]=0
#        else: fwt2=fwt1
    
def GS_modified_test(sigma_i,sigma_s,Rho,AS,AI,AP,include_noise=True,Nsigma=[3,3],Nsamples=[2**7,2**7],Nphotons=300,max_iter=200):
    '''
    Run GS Algorithm many times varying the parameters above to test
    performance.
    '''
    fid=np.zeros((Rho.size,2))
    errors=np.zeros((Rho.size,2))
    print('rho,Ai,As,Ap')
    for k in range(Rho.size):
        print(Rho[k],AI[k],AS[k],AP[k])
    
        Wi,Ws,Fww=gaussian_state(wi0,ws0,sigma_i,sigma_s,Rho[k],
                               As=AS[k],Ai=AI[k],Ap=AP[k],Nsigma=Nsigma,Nsamples=Nsamples)
        Ti,Ts,fwt=fft_state_1D(Wi,Ws,Fww,axis=0)
        Ti,Ts,ftw=fft_state_1D(Wi,Ws,Fww,axis=1)
        Ti,Ts,ftt=fft_state_2D(Wi,Ws,Fww)
    
        if include_noise:
            Iww=np.double(np.random.poisson(Nphotons*np.abs(Fww)**2))
            Iwt=np.double(np.random.poisson(Nphotons*(np.abs(fwt)/np.max(np.abs(fwt)))**2))
            Itw=np.double(np.random.poisson(Nphotons*(np.abs(ftw)/np.max(np.abs(ftw)))**2))
            Itt=np.double(np.random.poisson(Nphotons*(np.abs(ftt)/np.max(np.abs(ftt)))**2))
        else:
            Iww=np.double(np.abs(Fww)**2)
            Iwt=np.double((np.abs(fwt)/np.max(np.abs(fwt)))**2)
            Itw=np.double((np.abs(ftw)/np.max(np.abs(ftw)))**2)
            Itt=np.double((np.abs(ftt)/np.max(np.abs(ftt)))**2)         
    
        output=GS_modified(Iww,Itt,Itw,Iwt, 
                    initial_phase_guess=random_phase(Iww),max_iter=max_iter,error_model='NRMS')
        
        i,Fk,fk,errors[k,:]=output
    
        fid[k,0]=fidelity(Fww,Fk,(Wi[0,:],Ws[:,0]))
        fid[k,1]=fidelity(ftt,fk,(Ti[0,:],Ts[:,0]))
    
    return errors,fid


def rect_ycut(X,fract=0.5):
    """
    Return a copy of X with elements to the right of the cut zeroed.
    """
    n=np.array(X.shape)
    X[:,np.int(np.floor(fract*n[1])):-1]=0
    return X


def GS_algorithm(intF,intf,initial_guess=np.array([]),max_iter=200,alpha=0,
                 error_model='NRMS',return_errors=False,
                 target_state=np.array([]),x=np.array([]),y=np.array([])):
    
    '''
    Gerchberg-Saxton algorithm for finding the phase from the modulous of 
    the joint spectral and joint temporal intensities.
    '''
    intf=np.double(intf)
    intF=np.double(intF)
    
    #Check that normalization of modulous data satifies Parseval's theorem
    intf*=np.sum(intF)/np.sum(intf)
    
    #Initialize function
    if initial_guess.size!=0:
        Fk=initial_guess
    else:
        Fk=np.sqrt(intF)
    
    #Initialize iteration
    EF=100
    Ef=100
    
    if return_errors:
        Errors=np.zeros((max_iter,2))
        
    if target_state.size!=0:
        fid=np.zeros((max_iter,2))
    
    i=0
    while (EF>1e-5 or Ef>1e-5):
        if i>=max_iter:
            i-=1
            #print 'Maximum iterations reached'
            break
        
        #Calculate joint temporal amplitude
        fk1=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Fk),norm='ortho'))
        #fk=(np.fft.ifft2(Fk))
        
        if np.isnan(fk1).any(): return i,Fk1,fk1,Fk2,fk2,Errors,fid
        
        #Replace modulous with measured temporal values
        #if i%10==0:print i           
        
        if 0 and (i>40) and (i<50):
            #fk2=fk1/(np.abs(fk1))*(np.sqrt(intf)+alpha*(np.sqrt(intf)-np.abs(fk1)))
            
            fk2=rect_ycut(fk1/(np.abs(fk1)),0.49)
            #fk2*=np.sum(intf)/np.sum(np.abs(fk2)**2)
            
        else:
            fk2=fk1/(np.abs(fk1)+1e-4*np.max(np.abs(fk1)))*np.sqrt(intf)
            #fk2=fk1/(np.abs(fk1))*np.sqrt(intf)
            
        #Calculate joint spectral amplitude
        Fk1=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(fk2),norm='ortho'))
        
        #Replace modulous with measured spectral values
        Fk2=Fk1/(np.abs(Fk1)+1e-4*np.max(np.abs(Fk1)))*np.sqrt(intF)
        #Fk2=np.real(Fk2)+1j*np.abs(np.imag(Fk2))

       # Fk2=Fk1/(np.abs(Fk1))*np.sqrt(intF)
        
        #Calculate error
        if error_model=='NRMS':
            #print np.isnan(intF).any(),np.isnan(Fk1).any()
            EF,muF=NRMS(np.sqrt(intF),np.abs(Fk1))
            Ef,muf=NRMS(np.sqrt(intf),np.abs(fk1))
        elif error_model=='Chi2':
            EF,muF=chi2_reduced(intF,np.abs(Fk1)**2)
            Ef,muf=chi2_reduced(intf,np.abs(fk1)**2)
        elif error_model=='FROG':
            EF,muF=FROG_error(intF,np.abs(Fk1)**2)      
            Ef,muf=FROG_error(intf,np.abs(fk1)**2)
        
        if return_errors:
            Errors[i,:]=np.array([EF,Ef])
        
        if target_state.size!=0:
            fid[i,0]=fidelity(target_state,Fk1,(x,y))
            fid[i,1]=fidelity(np.conjugate(target_state),Fk1,(x,y))
            
        Fk=Fk2
        i+=1
    
    if return_errors and target_state.size!=0:
        return i+1,Fk1,fk1,Errors,fid
    elif return_errors:
        return i+1,Fk1,fk1,Fk2,fk2,Errors
    elif target_state.size!=0:
        return i+1,Fk1,fk1,np.array([EF,Ef]),fid
    else:
        return i+1,Fk1,fk1,np.array([EF,Ef])


def print_GS_output(intF,intf,output):
    i,Fk1,fk,Fk2,fk2,errors=output
    print('iterations',i)
    
    EF,muF=NRMS(np.sqrt(intF),np.abs(Fk1))      
    Ef,muf=NRMS(np.sqrt(intf),np.abs(fk))
    
    print('\nNRMS errors:', EF,Ef)
    print('Weight mu:',muF,muf)
    
    GF,muF=FROG_error(intF,np.abs(Fk1)**2)      
    Gf,muf=FROG_error(intf,np.abs(fk)**2)
    print('\nG error:',GF,Gf)
    print('Weight:',muF,muf)
     
    chi2F,nF=chi2_reduced(intF,np.abs(Fk1)**2)
    chi2f,nf=chi2_reduced(intf,np.abs(fk)**2)
       
    print('\nChi2:',chi2F,chi2f)
    print('Weight:',nF,nf)

        
def GS_test(wi0,ws0,sigma_i,sigma_s,Rho,AS,AI,AP,include_noise=True,Nsigma=[3,3],Nsamples=[2**7,2**7],Nphotons=300,max_iter=200):
    '''
    Run GS Algorithm many times for a Gaussian state varying the parameters above to test
    performance.
    '''
    fid=np.zeros((Rho.size,2))
    errors=np.zeros((Rho.size,2))
    print('rho,Ai,As,Ap')
    for k in range(Rho.size):
        print(Rho[k],AI[k],AS[k],AP[k])
    
        Wi,Ws,Fww=gaussian_state(wi0,ws0,sigma_i,sigma_s,Rho[k],
                               As=AS[k],Ai=AI[k],Ap=AP[k],Nsigma=Nsigma,Nsamples=Nsamples)
        Ti,Ts,ftt=fft_state_2D(Wi,Ws,Fww)
    
        if include_noise:
            Iww=np.double(np.random.poisson(Nphotons*np.abs(Fww)**2))
            Itt=np.double(np.random.poisson(Nphotons*(np.abs(ftt)/np.max(np.abs(ftt)))**2))
        else:
            Iww=np.double(np.abs(Fww)**2)
            Itt=np.double((np.abs(ftt)/np.max(np.abs(ftt)))**2)         
    
        output=GS_algorithm(Iww,Itt, 
                    initial_guess=np.sqrt(Iww)*random_phase(Iww),max_iter=max_iter,error_model='NRMS')
        
        i,Fk,fk,errors[k,:]=output
    
        fid[k,0]=fidelity(Fww,Fk,(Wi[0,:],Ws[:,0]))
        fid[k,1]=fidelity(ftt,fk,(Ti[0,:],Ts[:,0]))
    
    return errors,fid
    
    
    
