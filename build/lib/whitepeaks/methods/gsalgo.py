'''
###############################################################################
GERCHBERG SAXTON ALGORITHM MODULE
###############################################################################

Routines in this module:

> GS_1D(Iw,It)
> GS_modified(Iww,Itt,Itw,Iwt)
> run_gs_algorithm(gs_input,max_iter)
> GS_modified_test(sigma_i,sigma_s,Rho,AS,AI,AP)
> GS_algorithm(intF,intf)
> print_GS_output(intF,intf,output)
> GS_test(wi0,ws0,sigma_i,sigma_s,Rho,AS,AI,AP)

'''

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import entropy 
from skimage.restoration import unwrap_phase
from matplotlib.colors import LinearSegmentedColormap

from ..analytics import *
from ..interface import *
from ..states import *

cmap_darkcyan=LinearSegmentedColormap.from_list('my_colormap',['black','darkcyan','white'])
cmap_dodgerblue=LinearSegmentedColormap.from_list('my_colormap',['black','dodgerblue','white'])

colormaps=['gist_heat',cmap_darkcyan,cmap_darkcyan,cmap_dodgerblue]
colormaps=['jet','jet','jet','jet']
colormaps=['viridis','viridis','viridis','viridis']

def GS_1D(Iw,It,
          initial_guess='random',max_iter=5000,tol=1e-5,
          error_model='NRMS',return_errors=False,method='modified',
          target_state=np.array([]),x=np.array([])):
    '''
    Gerchberg-Saxton algorithm for finding 1D phase from the spectral and
    temporal intensity. 

    Parameters
    ----------
    Iw: ndarray 
        1D array of Frequency data.     
    It: ndarray 
        1D array of Time data.     
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
        fw:ndarray
           2D array of complex frequency-frequency amplitudes.
        ft:ndarray
            2D array of complex time-time amplitudes.
        Errors:1D array,optional 
            Error at each iteration. Only included in output if
            return_errors is True.
        fid: 1D array,optional
            Fidelity at each iteration. Only included in output if target
            state is specified.  
    '''

    Iw=np.double(Iw)
    It=np.double(It)
    
    #Check that normalization of modulus data satifies Parseval's theorem
    It*=np.sum(Iw)/np.sum(It)

    #Initialize function
    if isinstance(initial_guess,str):
        if initial_guess=='random':
            Fw=np.sqrt(Iw)*random_phase(Iw)
        elif initial_guess=='flat': 
            Fw=np.sqrt(Iw)
    else: 
        Fw=initial_guess

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
        ft1=np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Fw),norm='ortho'))
        
        #Replace modulous with measured spectral values
        ft2=ft1/(np.abs(ft1)+1e-4*np.max(np.abs(ft1)))*np.sqrt(It)

        #Calculate joint spectral amplitude
        fw1=np.fft.fftshift(np.fft.fft(np.fft.fftshift(ft2),norm='ortho'))

        #Replace modulous with measured spectral values
        fw2=fw1/(np.abs(fw1)+1e-4*np.max(np.abs(fw1)))*np.sqrt(Iw)

        #Calculate error
        if error_model=='NRMS':
            #print np.isnan(Iww).any(),np.isnan(Fk1).any()
            EF,muF=NRMS(np.sqrt(Iw),np.abs(fw1))
            Ef,muf=NRMS(np.sqrt(It),np.abs(ft1))
        elif error_model=='FROG':
            #USE FROG ERROR
            EF,muF=FROG_error(Iw,np.abs(fw1)**2)      
            Ef,muf=FROG_error(It,np.abs(ft1)**2)
        elif error_model=='KL':
            #USE KL divergence or relative entropy
            EF=np.sum(entropy(Iw,np.abs(fw1)**2))
            Ef=np.sum(entropy(It,np.abs(ft1)**2))
        
        if return_errors:
            Errors[i,:]=np.array([EF,Ef])
        
        if target_state.size!=0:
            fid[i,0]=fidelity(target_state,fw1,x)
            fid[i,1]=fidelity(np.conjugate(target_state),fw1,(x))
            
        Fw=fw2
        i+=1
    
    if return_errors and target_state.size!=0:
        return i+1,fw1,ft1,Errors,fid
    elif return_errors:
        return i+1,fw1,ft1,Errors
    elif target_state.size!=0:
        return i+1,fw1,ft1,fid
    else:
        return i+1,fw1,ft1

def GS_modified(Iww,Itt,Itw,Iwt,
                initial_guess='random',max_iter=5000,tol=1e-5,
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
    method: {'modified','gs','full'}, optional
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
        elif method=='full':
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
        elif method=='full':
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


def run_gs_algorithm(gs_input,max_iter,
                     method='modified',error_model='FROG',
                     initial_guess='random',tol=1E-5,
                     target_state=[],
                     verbose=2):
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
    method: {'modified','gs','full'}, optional
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
    verbose:int
        Define level of printed output, i.e.,
        0: No console output and no plots
        1: Console output and no plots
        2: Console output and plots

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
    
    if verbose > 0:
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

        N=F.shape
        F*=np.exp(-1j*np.angle(F[N[0]//2,N[1]//2]))
        Fk_angle=np.angle(F)
        Fk_angle[np.where((np.abs(F)**2/np.max(np.abs(F)**2))<=np.exp(-(2.5)**2/2))]=-np.pi
        
        if verbose > 1:
            fig,ax=plt.subplots(1,3,figsize=(12,3))
            ax=ax.reshape(-1)
            S0=ax[0].pcolormesh(X,Y,I,cmap=colormaps[k])
            S1=ax[1].pcolormesh(X,Y,np.abs(F)**2/np.max(np.abs(F)**2),cmap=colormaps[k])
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
    
    if verbose > 1:
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
    
    if verbose > 0:
        print('FROG',FROG_error(Iww,np.abs(Fk)**2)[0],FROG_error(Itt,np.abs(fk)**2)[0])
        print('NRMS',NRMS(np.sqrt(Iww),np.abs(Fk))[0],NRMS(np.sqrt(Itt),np.abs(fk))[0])
        print('KL',np.sum(entropy(Iww,np.abs(Fk)**2)),np.sum(entropy(Itt,np.abs(fk)**2)))

        if len(target_state)==3:
            #If fidelity was calculated
            print('Fidelity',fid[-1,:])

    return output

def GS_modified_test(sigma_i,sigma_s,Rho,AS,AI,AP, A3I=0,A3S=0,
                     method='full',include_noise=True,Nsigma=[3,3],
                     Nsamples=[2**7,2**7],Nphotons=300,max_iter=200):
    '''
    Run modified GS algorithm many times varying the parameters above to test
    performance. Rho, AS, AI, AP must be the same size.

    Parameters
    ----------
    sigma_i:float
        Idler frequency bandwidth
    sigma_s:float
        Signal frequency bandwidth
    Rho:ndarray
        Array of correlation values to test.
    AS:ndarray
        Array of signal chirp values to test.
    AI:ndarray
        Array of idler chirp values to test.
    AP:ndarray
        Array of pump chirp values to test.
    method:{'modified','gs','full'}:
        Method to use in GS algorithm. See modified_GS function.
    include_noise:bool,optional
        Set to 'True' to add Poissonian noise.
    Nsamples:list (float,float)
        Number of sample points along x and y in the grid.
    Nsigma:list (float,float)
        Size of the grid specfied a multiplicative factor of the standard
        deviation along x and y.
    include_noise:bool,optional
        Include Poisson noise in simulation.
    Nphotons:
        Number of photons to include in Poissonian noise model.
    max_iter: int,optional
        Maximum number of loops in algorithm

    Returns
    ---------- 
    output: tuple with following variables
        errors:1D array
            Error between ideal and reconstructed state. 
        fid: 1D array
            Fidelity between ideal and reconstructed state. 
            '''
    fid=np.zeros((Rho.size,2))
    errors=np.zeros((Rho.size,2))
    print('rho,Ai,As,Ap')
    for k in range(Rho.size):
        print(Rho[k],AI[k],AS[k],AP[k])
    
        Wi,Ws,Fww=gaussian_state(wi0,ws0,sigma_i,sigma_s,Rho[k],
                               As=AS[k],Ai=AI[k],Ap=AP[k],Nsigma=Nsigma,Nsamples=Nsamples)

        if isinstance(A3S,np.ndarray): Fww=Fww*np.exp(1j*A3S[k]*(Ws-ws0)**3)
        if isinstance(A3I,np.ndarray): Fww=Fww*np.exp(1j*A3I[k]*(Wi-wi0)**3)

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
                            initial_phase_guess=random_phase(Iww),method=method,
                            max_iter=max_iter,error_model='FROG')
        
        i,Fk,fk,errors[k,:]=output
    
        fid[k,0]=fidelity(Fww,Fk,(Wi[0,:],Ws[:,0]))
        fid[k,1]=fidelity(ftt,fk,(Ti[0,:],Ts[:,0]))
    
    return errors,fid

def GS_algorithm(intF,intf,
                 initial_guess=np.array([]),max_iter=200,alpha=0,
                 error_model='NRMS',return_errors=False,
                 target_state=np.array([]),x=np.array([]),y=np.array([])):
    
    '''
    Gerchberg-Saxton algorithm for finding the phase from the modulus of 
    the joint spectral and joint temporal intensities.

    Parameters
    ----------
    intF: ndarray 
        2D array of Frequency-Frequency count data.     
    intf: ndarray 
        2D array of Time-Time data.     
    max_iter: int,optional
        Maximum number of loops in algorithm
    initial_guess:ndarray,optional
        Specify initial guess. Otherwise, intial guess is chosen
        automatically.
   return_errors:bool,optional
        Set to 'True' to return errors at each iteration.
   error_model: {'FROG', 'NRMS', 'KL'} str, optional
        Option to use FROG, Normalized root-mean square Kullback-Leibler
        divergence to calculate error between reconstructed and measured
        state.
    target_state: tuple (Wi,Ws,F), optional 
        Target state for fidelity calculation.

    Returns
    ---------- 
    out: tuple with following variables
        i:  int
            Number of iterations
        fww:ndarray
            2D array of complex frequency-frequency amplitudes.
        ftt:ndarray
            2D array of complex time-time amplitudes.
        Errors:1D array,optional 
            Error at each iteration. Only included in output if
            return_errors is True.
        fid: 1D array,optional
            Fidelity at each iteration. Only included in output if target
            state is specified.  

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
        fk2=fk1/(np.abs(fk1)+1e-4*np.max(np.abs(fk1)))*np.sqrt(intf)
            
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
    '''
    Print values of different error models to compare. 

    Parameters
    ----------
    intF: ndarray 
        2D array of Frequency-Frequency count data.     
    intf: ndarray 
        2D array of Time-Time data.     
    output:tuple(integer, 4xndarray)
        Output of GS_algorithm (i,Fk1,fk,Fk2,fk2,errors)

    '''
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

        
def GS_test(wi0,ws0,sigma_i,sigma_s,Rho,AS,AI,AP,
            A3I=0,A3S=0,
            method='full',include_noise=True,max_iter=500,
            Nsigma=[3,3],Nsamples=[2**7,2**7],Nphotons=300):
    '''
    Run GS Algorithm many times for a Gaussian state varying the parameters above to test
    performance.

    !!!
    This function looks the same as GS_modified_test. Delete it?
    !!!
    '''
    fid=np.zeros((Rho.size,2))
    errors=np.zeros((Rho.size,2))
    print('k,rho,Ai,As,Ap')
    for k in range(Rho.size):
        print('\r',k,'of',Rho.size,Rho[k],AI[k],AS[k],AP[k],end='')
    
        Wi,Ws,Fww=gaussian_state(wi0,ws0,sigma_i,sigma_s,Rho[k],
                               As=AS[k],Ai=AI[k],Ap=AP[k],Nsigma=Nsigma,Nsamples=Nsamples)


        if isinstance(A3S,np.ndarray): Fww=Fww*np.exp(1j*A3S[k]*(Ws-ws0)**3)
        if isinstance(A3I,np.ndarray): Fww=Fww*np.exp(1j*A3I[k]*(Wi-wi0)**3)

        

        Ti,Ts,ftt=fft_state_2D(Wi,Ws,Fww)
        Wi,Ws,Ti,Ts,Iww,Itw,Iwt,Itt= create_tf_data((Wi,Ws,Fww),add_poisson_noise=include_noise)
    
        output=GS_modified(Iww,Itt,Itw,Iwt,method=method, 
                    initial_guess=random_phase(Iww),max_iter=max_iter,error_model='FROG',return_errors=True)

        i,Fk,fktw,fkwt,fk,Errors_array=output

        errors[k,:]=Errors_array[-1,:]
    
        fid[k,0]=fidelity(Fww,Fk,(Wi[0,:],Ws[:,0]))
        #fid[k,1]=fidelity(Fww,np.conjugate(Fk),(Wi[0,:],Ws[:,0]))
        fid[k,1]=fidelity(ftt,fk,(Ti[0,:],Ts[:,0]))
    
    return errors,fid
