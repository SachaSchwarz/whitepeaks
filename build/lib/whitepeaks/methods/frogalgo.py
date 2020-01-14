'''
###############################################################################
FROG ALGORITHM MODULE
###############################################################################

Routines in this module:

> CalcEsig(E1,E2)
> CalcEsigOuter(E1,E2)
> CalcOuter(Esig)
> vanilla_FROG(spectrogram)
> PCGP_FROG(spectrogram)
> MPCGP_FROG(spectrogram)
> run_PCGP_FROG(spectrogram_data)

'''

import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

from ..analytics import *
from ..interface import *
from ..states import *

def CalcEsig(E1,E2,
             nonlinearity='SHG'):
    '''
    Calculate the upconverted signal field for FROG or XFROG. Specify the
    nonlinearity.

    Parameters
    ----------
    E1: 1D array
       E field in time 
    E2: 1D array
        Gate field in time
    
    Returns
    ----------
    out: 2D array
        Esig(t,tau)

    '''
    if E1.shape!=E2.shape:
        print('Error: E1 and E2 must have the same shape.') 

    n=E1.size
    Esigpad=np.zeros((2*n,2*n),dtype='complex')

    Ekp1=np.pad(E1,n//2,'constant')
    Ekp2=np.pad(E2,n//2,'constant')

    if nonlinearity=='SHG':
        for i in range(2*n):
            Esigpad[:,i]=Ekp1*np.roll(Ekp2,n+i)
        
    return Esigpad[n//2:-n//2,n//2:-n//2]

def CalcEsigOuter(E1,E2):
    '''
    Calculate the upconverted signal field Esig(t,tau) as a function of
    delay tau for FROG or XFROG using the outer product, row rotations,
    and column manipulations. 

    See Trebino Ch. 21, pages 362-363.

    Parameters
    ----------
    E1: 1D array
       E field in time 
    E2: 1D array
        Gate field in time
    
    Returns
    ----------
    out: 2D array
        Esig(t,tau)
        
    '''

    n=E1.size

    #Calculate outer product
    O=np.outer(E1,E2)

    #Rotate the columns of the outer product
    for i in range(n):
        O[i,:]=np.roll(O[i,:],-i)

    #Roll and reverse columns to create Esig
    Esig=np.fliplr(np.roll(O,n//2,axis=1))

    #Return the signal field
    return Esig

def CalcOuter(Esig):
    '''
    Calculate the outer product from the upconverted signal field Esig(t,tau). 

    Parameters
    ----------
    Esig: ndarray
        2D array Esig(t,tau)
    
    Returns
    ----------
    out: ndarray
        2D array Outer(t,tau)
    '''

    n=np.shape(Esig)[0]

    #Reverse and roll columns to create the outer product 
    O=np.roll(np.fliplr(Esig),-n//2,axis=1)

    #Rotate the columns of the outer product
    for i in range(n):
        O[i,:]=np.roll(O[i,:],i)

    return O

def vanilla_FROG(spectrogram,
                 max_iter=500,
                 initial_guess='random',tol=1e-5, error_model='FROG'):
    '''
    Basic "Vanilla" FROG algorithm for reconstructing amplitude and phase
    of the electric field. Only works for SHG FROG.
    
    See Trebino-FROG Chapter 8 page 159.

    Parameters
    ----------
    spectrogram: 2D array
        Spectrogram[tau,omega] where tau is the pulse delay and omega is
        the measured upconverted frequency.
    max_iter: int,optional
        Maximum number of loops in algorithm
    initial_guess:{'random','flat', 1darray}
    tol:float,optional 
        stop iterations when error<tol.
    error_model: {{'FROG'}, 'None'}, optional
        Method used to calculate error in reconstruction.

    Returns
    ----------
    out: tuple with following variables
        k:  int
            number of iterations
        Ek: 1D array
            Reconstructed field  in time
        Sk1:2D array
            Reconstructed spectrogram 
        EFk:1D array 
            FROG error at each iteration.

    '''

    S=spectrogram

    n=S.shape[0]

    if initial_guess=='flat':
        Eguess=np.sqrt(S).sum(0)/np.sqrt(S).sum()

    elif initial_guess=='random':
        Eguess=np.sqrt(S.sum(0)/S.sum())
        Eguess=Eguess*random_phase(Eguess)
    else:
        Eguess=initial_guess

    EF=100 

    #Specify initial guess
    Ek=Eguess

    k=0
    while (EF>tol):
        if k>=max_iter:
            k-=1
            #print('Maximum iterations reached')
            break

        #Calculate the signal field for the kth iteration 
        Esig=CalcEsig(Ek,Ek)

        #Calculate the spectrogram of the signal field
        Sk1=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Esig,axes=0),axes=[0],norm='ortho'),axes=0)

        #Replace amplitude of the spectrogram with the measured spectrogram
        Sk2=Sk1/(np.abs(Sk1)+1e-4*np.max(np.abs(Sk1)))*np.sqrt(S)

        #Fourier transform 
        Esig1=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Sk2,axes=0),norm='ortho',axes=[0]),axes=0)

        #Sum over delay tau (I think this only works for FROG not XFROG, check.)
        Ek2=Esig1.sum(1)/np.max(np.abs(Esig1)**2)
        
        #Define kth+1 field
        Ek=Ek2

        #Calculate the FROG error between the measured and reconstructed spectrogram.
        if error_model=='FROG':
            EF,muF=FROG_error(S,np.abs(Sk1)**2)      
            
        k+=1

    return k, Ek, Sk1, EF

def PCGP_FROG(spectrogram,
              max_iter=500,
                 initial_guess='random',type='FROG',tol=1e-6,Egate=0,
              error_model='FROG',target_state=()):
    '''
    Principal Component Generalized Projections FROG Algorithm for
    reconstructing amplitude and phase of the electric field. 
    
    See Trebino Ch 21.

    Parameters
    ----------
    spectrogram: 2D array
        Spectrogram[tau,omega] where tau is the pulse delay and omega is
        the measured upconverted frequency.
    max_iter: int,optional
        Maximum number of loops in algorithm
    initial_guess:{'random','flat',1darray}
    type: {'FROG','XFROG','BlindFROG'}, optional
        For XFROG, the gate field parameter Egate must be specified.
    tol:float,optional 
        stop iterations when error<tol.
    Egate: 1D array, optional
        Gate E field for XFROG. 
    error_model: {{'FROG'}, 'None'}, optional
        Method used to calculate error in reconstruction.
    target_state: tuple (t,Et), optional 
        Target state for fidelity calculation.

    Returns
    ----------
    out: tuple with following variables
        k:  int
            Number of iterations
        Pk: 1D array
            Reconstructed field  in time
        Gk: 1D array 
            Reconstructed gate in time
        Sk1:2D array
            Reconstructed spectrogram 
        EFk:1D array 
            FROG error at each iteration.
        fid: 1D array,
            Fidelity at each iteration. Only included in output if target
            state is specified.  
    '''

    if (type=='XFROG' and np.isscalar(Egate)):
        print('Please specify Egate field.')

    S=spectrogram

    n=S.shape[0]

    if isinstance(initial_guess,str):
        if initial_guess=='flat':
            Eguess=np.sqrt(S).sum(0)/np.sqrt(S).sum()

        elif initial_guess=='random':
            Eguess=np.sqrt(S.sum(0)/S.sum())
            Eguess=Eguess*random_phase(Eguess)
    else:
        Eguess=initial_guess

    EF=100
    EFk=np.zeros(max_iter)

    if len(target_state)!=0:
        fid=np.zeros(max_iter)
        ttarget,Etarget=target_state

    #Specify initial guess for the field Pk and the gate Gk.
    Pk=Eguess
    Gk=Eguess

    k=0
    while (EF>tol):
        if k>=max_iter:
            #k-=1
            #print('Maximum iterations reached')
            break

        #Calculate the signal field for the kth iteration using the outer
        #product method.
        if type=='FROG':
            #For FROG the two outer products must be equal.
            Esig=CalcEsigOuter(Pk,Gk)+CalcEsigOuter(Gk,Pk)
        elif type=='XFROG':
            #For XFROG we assume we know the gate pulse. Could compare Gk
            #with the input gate as a consistency check. 
            Esig=CalcEsigOuter(Pk,Egate)
        elif type=='BlindFROG':
            #For Blind FROG, we don't know the signal or the gate. 
            Esig=CalcEsigOuter(Pk,Gk)
        else: 
            print('Please specify FROG type.')
            break

        #Calculate the spectrogram of the signal field
        Sk1=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Esig,axes=0),axes=[0],norm='ortho'),axes=0)

        #Apply intensity contraint. 
        #Replace amplitude of the spectrogram with the measured spectrogram.
        Sk2=Sk1/(np.abs(Sk1)+1e-4*np.max(np.abs(Sk1)))*np.sqrt(S)

        #Fourier transform back to time domain.
        Esig1=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Sk2,axes=0),norm='ortho',axes=[0]),axes=0)

        #Calculate Outer Product from the signal field
        Outerk=CalcOuter(Esig1)

        #Calculate the principal component of the field and the gate.
        Pk2=np.matmul(Outerk @ Outerk.T, Pk)
        Gk2=np.matmul(Outerk.T @ Outerk, Gk)

        #Define kth+1 fields
        Pk=np.array(Pk2)/np.max(np.abs(Pk2))
        Gk=np.array(Gk2)/np.max(np.abs(Gk2))

        #Calculate the FROG error between the measured and reconstructed spectrogram.
        if error_model=='FROG':
            #print(Sk1)
            EF,muF=FROG_error(S,np.abs(Sk1)**2)      
            EFk[k]=EF
        #If input state is known, calculate fidelity with input state.
        if len(target_state)!=0:
            fid[k]=fidelity(Etarget,Pk,ttarget)
            
        k+=1

    if len(target_state)!=0:
        return k, Pk, Gk, Sk1, EFk[:k],fid[:k]   
    else:
        return k, Pk, Gk, Sk1, EFk[:k]

def MPCGP_FROG(spectrogram,
               max_iter=500,
               initial_guess='random',type='FROG',number_of_instances=5,tol=1e-6,Egate=0,
              error_model='FROG',target_state=()):
    '''
    Mixed principal Component Generalized Projections FROG Algorithm for
    reconstructing amplitude and phase of the electric field. 
    
    See Trebino Ch 21.

    Parameters
    ----------
    spectrogram: 2D array
        Spectrogram[tau,omega] where tau is the pulse delay and omega is
        the measured upconverted frequency.
    max_iter: int,optional
        Maximum number of loops in algorithm
    initial_guess:{'random','flat',1darray}
    type: {'FROG','XFROG','BlindFROG'}, optional
        For XFROG, the gate field parameter Egate must be specified.
    tol:float,optional 
        stop iterations when error<tol.
    Egate: 1D array, optional
        Gate E field for XFROG. 
    error_model: {{'FROG'}, 'None'}, optional
        Method used to calculate error in reconstruction.
    target_state: tuple (t,Et), optional 
        Target state for fidelity calculation.

    Returns
    ----------
    out: tuple with following variables
        k:  int
            Number of iterations
        Pk: 1D array
            Reconstructed field  in time
        Gk: 1D array 
            Reconstructed gate in time
        Sk1:2D array
            Reconstructed spectrogram 
        EFk:1D array 
            FROG error at each iteration.
        fid: 1D array,
            Fidelity at each iteration. Only included in output if target
            state is specified.  
    '''

    if (type=='XFROG' and np.isscalar(Egate)):
        print('Please specify Egate field.')

    S=spectrogram

    n=S.shape[0]

    if isinstance(initial_guess,str):
        if initial_guess=='flat':
            Eguess=np.sqrt(S).sum(0)/np.sqrt(S).sum()

        elif initial_guess=='random':
            Eguess=np.sqrt(S.sum(0)/S.sum())
            Eguess=Eguess*random_phase(Eguess)
    else:
        Eguess=initial_guess

    EF=100
    EFk=np.zeros(max_iter)

    if len(target_state)!=0:
        fid=np.zeros(max_iter)
        ttarget,Etarget=target_state

    #Specify initial guess for the field Pk and the gate Gk.
    Pk=np.zeros((Eguess.size,number_of_instances),dtype='complex')
    Gk=np.ones(Pk.shape,dtype='complex')
    Pk2,Gk2=Pk.copy(),Gk.copy()
    Esig=np.zeros((spectrogram.shape[0],spectrogram.shape[1],number_of_instances),dtype='complex')
    Sk1,Sk2=Esig.copy(),Esig.copy()
    
    x=np.arange(Eguess.size)
    sigma=10
    for i in range(number_of_instances):
        #Make initial guess a mixture of Hermite-Gauss Polynomials.
        Pk[:,i]=np.exp(-(x-x.size/2)**2/(2*sigma**2))*(
        np.polynomial.hermite.hermval((x-x.size/2)/sigma,np.identity(number_of_instances)[i]))
        
    k=0
    while (EF>tol):
        print("\r",k,"of",max_iter,end="")
        if k>=max_iter:
            #k-=1
            #print('Maximum iterations reached')
            break

        #Calculate the signal field for the kth iteration using the outer
        #product method.
        if type=='FROG':
            #For FROG the two outer products must be equal.
            for i in range(number_of_instances):
                Esig[:,:,i]=CalcEsigOuter(Pk,Gk)+CalcEsigOuter(Gk,Pk)
        elif type=='XFROG':
            #For XFROG we assume we know the gate pulse. Could compare Gk
            #with the input gate as a consistency check. 
            for i in range(number_of_instances):
                Esig[:,:,i]=CalcEsigOuter(Pk[:,i],Egate)
        elif type=='BlindFROG':
            #For Blind FROG, we don't know the signal or the gate. 
            for i in range(number_of_instances):
                Esig[:,:,i]=CalcEsigOuter(Pk,Gk)
        else: 
            print('Please specify FROG type.')
            break

        #Calculate the spectrogram of the signal field
        for i in range(number_of_instances):
            Sk1[:,:,i]=np.fft.fftshift(np.fft.fft2(
                np.fft.fftshift(Esig[:,:,i],axes=0),axes=[0],norm='ortho'),axes=0)

        #Apply intensity contraint. 
        #Replace amplitude of the spectrogram with the measured spectrogram.
        for i in range(number_of_instances):
            Sk2[:,:,i]=Sk1[:,:,i]*np.sqrt(
                S/(np.sum(np.abs(Sk1)**2,axis=2)+1e-4*np.max(np.sum(np.abs(Sk1)**2,axis=2))))

        #Fourier transform back to time domain.
        for i in range(number_of_instances):
            Esig1=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Sk2[:,:,i],axes=0),norm='ortho',axes=[0]),axes=0)

            #Calculate Outer Product from the signal field
            Outerk=CalcOuter(Esig1)

            #Calculate the principal component of the field and the gate.
            Pk2[:,i]=np.matmul(Outerk @ Outerk.T, Pk[:,i])
            Gk2[:,i]=np.matmul(Outerk.T @ Outerk, Gk[:,i])

            #Define kth+1 fields
            Pk[:,i]=np.array(Pk2[:,i])/np.max(np.abs(Pk2[:,i]))
            Gk[:,i]=np.array(Gk2[:,i])/np.max(np.abs(Gk2[:,i]))

        #Calculate the FROG error between the measured and reconstructed spectrogram.
        if error_model=='FROG':
            #print(Sk1)
            EF,muF=FROG_error(S,np.sum(np.abs(Sk1)**2,axis=2))      
            EFk[k]=EF
        #If input state is known, calculate fidelity with input state.
        if len(target_state)!=0:
            fid[k]=fidelity(Etarget,Pk,ttarget)
            
        k+=1

    if len(target_state)!=0:
        return k, Pk, Gk, Sk1, EFk[:k],fid[:k]   
    else:
        return k, Pk, Gk, Sk1, EFk[:k]

def run_PCGP_FROG(spectrogram_data,
                  max_iter=500,
                 initial_guess='random',type='FROG',tol=1e-6,Egate=0,
                  error_model='FROG',target_state=(),plots=True):
    '''
    Run the Principle Component Generalized Projections FROG Algorithm for
    and make plots.

    Parameters
    ----------
    spectrogram_data: tuple (2D array, 2D array, 2D array)
        (Time, Frequency, Spectrogram)
    max_iter: int,optional
        Maximum number of loops in algorithm
    initial_guess:{'random','flat',1darray}
    type: {'FROG','XFROG','BlindFROG'}, optional
    tol:float,optional 
        stop iterations when error<tol.
    Egate: 1D array, optional
        Gate E field for XFROG. 
    error_model: {{'FROG'}, 'None'}, optional
        Method used to calculate error in reconstruction.
    target_state: tuple (t,Et), optional 
        Target state for fidelity calculation.
    plots: bool, optional
        Set to False to prevent making plots.

    Returns
    ----------
    out:tuple(k,tk,Etk,wk,Ewk,Gk,Sk,EF)
    k:  int
        number of iterations
    tk: 1D array,
        Time array
    Etk: 1D array
        Reconstructed field in time.
    wk: 1D array,
        Frequency array
    Ewk: 1D array
        Reconstructed field in frequency.
    Gk: 1D array 
        Reconstructed gate in time.
    Sk: 2D array
        Reconstructed spectrogram 
    EFk:1D array 
        FROG error at each iteration.

    '''

    T,W,S=spectrogram_data

    #T,W=FROG_grid(2*w0,dw,axis=1,n=np.shape(S)[0])
    w0=(np.sum(W[:,0]*S.sum(1))/np.sum(S))/2

    frog_output=PCGP_FROG(S, max_iter=max_iter,
                          initial_guess=initial_guess, type=type,
                          tol=tol,Egate=Egate, error_model=error_model,
                          target_state=target_state)

    if len(target_state)!=0:
        k,Ek,Gk,Sk,EF,fid=frog_output
    else:
        k,Ek,Gk,Sk,EF=frog_output

    print('Iterations',k) 
    print('FROG Error', EF[-1]) 

    tk,Etk=T[0,:],Ek
    wk,Ewk=fft_pulse(tk,Etk,x0=w0)

    if plots:

        #PLOT FROG OUTPUT 
        #FROG ERROR
        fig,ax=plt.subplots(1,1,figsize=(10,2))
        ax.semilogy(EF)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('FROG Error')
        plt.show()
        
        if len(target_state)!=0:
            print('Fidelity',fid[-1]) 
            fig,ax=plt.subplots(1,1,figsize=(10,2))
            ax.plot(fid)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Fidelity')
            plt.show()
        
        #MEASURED AND RECONSTRUCTED SPECTROGRAMS
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax[0].set_title('Measured')
        S0=ax[0].pcolormesh(T,W,S)
        ax[1].set_title('Reconstructed')
        S1=ax[1].pcolormesh(T,W,np.abs(Sk)**2)

        ax[0].set_xlabel(r'Delay $\tau$ (fs)')
        ax[0].set_ylabel(r'Frequency $\omega$ (fs$^{-1}$)')
        ax[1].set_xlabel(r'Delay $\tau$ (fs)')
        ax[1].set_ylabel(r'Frequency $\omega$ (fs$^{-1}$)')
        [fig.colorbar([S0,S1][i],ax=ax[i]) for i in range(2)]
        plt.tight_layout()
        plt.show()

        #PLOT RECONSTRUCTED PULSE

        fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax2=ax.copy()

        ax[0].plot(wk,np.abs(Ewk)**2)
        ax2[0]=ax[0].twinx()
        ax2[0].plot(wk,np.unwrap(np.angle(Ewk)),'--')

        ax[1].plot(tk,np.abs(Etk)**2,'C1')
        ax2[1]=ax[1].twinx()
        ax2[1].plot(tk,np.unwrap(np.angle(Etk)),'C1--')

        ax[0].set_title(r'$E(\omega)$')
        ax[0].set_xlabel(r'Frequency (fs$^-1$)')
        ax[0].set_ylabel(r'Amplitude')
        ax2[0].set_ylabel(r'Phase')

        ax[1].set_title(r'$E(t)$')
        ax[1].set_xlabel(r'Time (fs)')
        ax[1].set_ylabel(r'Amplitude')
        ax2[1].set_ylabel(r'Phase')
        plt.tight_layout()
        plt.show()
    
    #return frog_output
    return k,tk,Etk,wk,Ewk,Gk,Sk,EF
