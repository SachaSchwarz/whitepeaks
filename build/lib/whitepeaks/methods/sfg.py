'''
###############################################################################
SUM FREQUENCY GENERATION MODULE
###############################################################################

Routines in this module:

> G(x,a,x0,sigma_x,kx)
> model_upconversion_spectrogram(Wi,Ws,F_source,wg0,sigma_g)
> model_double_upconversion(Wi,Ws,F_source,wg0,sigma_g)
> sfg_phase_matching(W1,W3)
> sfg_phase_matching_withQ(W1,W3)
> dither_phase_matching(W1,W3)
> sfg_transfer_function(W1,W2,F_source,wg0,sigma_g)
> model_photon_sfg(W1,W2,F_source,T)
> fast_double_upconversion(Wi,Ws,F_source,wg0,sigma_g
> model_gaussian_sfg(w10,sigma_1,w20,sigma_2)

'''



########################## Model SFG for Optical Gating ####################### 
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import time

from scipy.optimize import minimize
from scipy.signal import fftconvolve

from ..analytics import *
from ..interface import *
from ..states import *

#Upconversion crystal
cr=crystal.BiBO()
angle=151.3

def G(x,a,x0,sigma_x,kx):
    return a*(np.exp(
        -((x-x0)**2.0/(2.0*sigma_x**2.0))-1j*kx*(x)))

def model_upconversion_spectrogram(Wi,Ws,F_source,wg0,sigma_g,
                                   Ag=0,axis=1,
                                   add_phase_matching=False,L=1e3,show_progress=True):
    '''
    Calculate expected nonlocal spectrogram by optically gating the first
    SPDC photon (Wi). Assumes a gaussian shape for the gate function. 

    Parameters
    ----------
    Wi: ndarray
        2D meshgrid of idler frequencies
    Ws: ndarray
        2D meshgrid of signal frequencies
    F_source: ndarray
        JSA of the source
    wg0: float
        Centre frequency of the gate
    sigma_g: float
        RMS bandwidth of the gate
    Ag: float
        Gate dispersion
    axis:int,optional
        Which axis to upconvert. Default axis=1 upconverts first photon. 
        To upconvert second photon, set axis=0.
    add_phase_matching: bool, optional
        Add phase matching to optical gating calculation.
    L: float,optional
        Upconversion crystal length in um.

    Returns
    ----------
    Ti: ndarray
        Meshgrid of idler gate delays.
    Ws: ndarray
        Meshgrid of signal frequencies.
    S2: ndarray
        Measured nonlocal spectrogram.

    '''
    if axis==0:
        #Upconvert second photon
        Wi,Ws,F_source=Ws.T,Wi.T,F_source.T

    wi0,ws0=get_moments(Wi,np.abs(F_source)**2)[0],get_moments(Ws.T,np.abs(F_source.T)**2)[0]

    #Define wavelengths and marginal bandwidths
    lambda_i0=w2f(wi0)  #idler
    lambda_g0=w2f(wg0)  #gate

    #FREQUENCY SPACING
    dw=np.diff(Wi[0,:])[0]

    #TEMPORAL RANGE
    Ti,Ts=fft_grid(Wi,Ws)
    tau_i=Ti[0,:]

    #UPCONVERTED FREQUENCIES
    w30=wi0+wg0
    lambda_30=w2f(w30)
    
    w3=(Wi[0,:]-wi0)+w30

    if add_phase_matching:
        #Calculate crystal axes parameters
        #Crystal Axes
        extra_ordinary=[angle,cr.ny,cr.nz] #Extra-ordinary
        ordinary=cr.nx #Ordinary 
        
        #GV and GVD
        dki,d2ki=1.0/GV(lambda_i0,cr.n_e,extra_ordinary),GVD(lambda_i0,cr.n_e,extra_ordinary) #idler
        dkg,d2kg=1.0/GV(lambda_g0,cr.n_e,extra_ordinary),GVD(lambda_g0,cr.n_e,extra_ordinary) #signal
        
        #GV and GVD
        dk3,d2k3=1.0/GV(lambda_30,cr.sellemeir,ordinary),GVD(lambda_30,cr.sellemeir,ordinary) #pump

        #print(dki,dkg,dk3)

    S2=np.zeros(Ti.shape)
    if show_progress: print(S2.shape) 

    for j in range(tau_i.size):

        if show_progress: print('\rIteration:',j,end='')
        S=np.zeros(Wi.shape)

        for i in range(w3.size):
            
            #Gate function
            G1=G(w3[i]-Wi,1,wg0,np.sqrt(2)*sigma_g,tau_i[j])

            #Add gate dispersion
            if Ag!=0:
                G1=G1*np.exp(1j*Ag*(Wi-wi0)**2)

            F_output=F_source*G1

            #Add phase-matching on the idler 
            if add_phase_matching:

                deltak_i=(dki*(Wi-wi0)+dkg*(w3[i]-Wi-wg0)-dk3*(w3[i]-w30)
                          +0.5*d2kg*(w3[i]-Wi-wg0)**2+0.5*d2ki*(Wi-wi0)**2
                          -0.5*d2k3*(w3[i]-w30)**2
                         )

                phi_i=np.pi*np.exp(-1j*deltak_i*L/2)*np.sinc(1/np.pi*deltak_i*L/2)
                
                F_output*=phi_i

            #Sum over Wi  
            S[:,i]=np.abs(np.trapz(F_output,dx=dw,axis=1))**2
            #if i==0: print('\r',time.clock()-starttime,end='')

        #Sum over w3
        S2[:,j]=np.trapz(S,dx=dw,axis=1)

    if axis==0:
        return Ws.T,Ti.T,S2.T
    else:
        return Ti,Ws,S2

def model_double_upconversion(Wi,Ws,F_source,wg0,sigma_g,
                              Ag=0, add_phase_matching=False,L=1e3,show_progress=True):
    '''
    Calculate expected joint temporal intensity by optically gating two
    SPDC photons. Assumes a gaussian shape for the gate function. This
    algorithm works but is quite slow. 

    Parameters
    ----------
    Wi: ndarray
        2D meshgrid of idler frequencies
    Ws: ndarray
        2D meshgrid of signal frequencies
    F_source: ndarray
        JSA of the source
    wg0: float
        Centre frequency of the gate
    Ag: float
        Gate dispersion
    sigma_g: float
        RMS bandwidth of the gate
    add_phase_matching: bool, optional
        Add phase matching to optical gating calculation.
    L: float,optional
        Upconversion crystal length in um.

    Returns
    ----------
    Ti: ndarray
        Meshgrid of idler gate delays.
    Ts: ndarray
        Meshgrid of singla gate delays.
    JTI: ndarray
        Measured joint temporal intensity

    '''
    
    #Get center wavelengths from F_source
    wi0,ws0=get_moments(Wi,np.abs(F_source)**2)[0],get_moments(Ws.T,np.abs(F_source.T)**2)[0]

    #Define wavelengths and marginal bandwidths
    lambda_i0=w2f(wi0)  #idler
    lambda_s0=w2f(ws0)  #signal
    lambda_g0=w2f(wg0)  #gate

    #FREQUENCY SPACING
    dw=np.diff(Wi[0,:])[0]

    #TEMPORAL RANGE
    Ti,Ts=fft_grid(Wi,Ws)
    tau_i=Ti[0,:]
    tau_s=Ts[:,0]

    #UPCONVERTED FREQUENCIES
    w30,w40=wi0+wg0,ws0+wg0
    lambda_30,lambda_40=w2f(w30),w2f(w40)
    
    w3=(Wi[0,:]-wi0)+w30
    w4=(Wi[0,:]-wi0)+w40

    if add_phase_matching:
        #Calculate crystal axes parameters
        #Crystal Axes
        extra_ordinary=[angle,cr.ny,cr.nz] #Extra-ordinary
        ordinary=cr.nx #Ordinary 
        
        #GV and GVD
        dki,d2ki=1.0/GV(lambda_i0,cr.n_e,extra_ordinary),GVD(lambda_i0,cr.n_e,extra_ordinary) #idler
        dks,d2ks=1.0/GV(lambda_s0,cr.n_e,extra_ordinary),GVD(lambda_s0,cr.n_e,extra_ordinary) #idler
        dkg,d2kg=1.0/GV(lambda_g0,cr.n_e,extra_ordinary),GVD(lambda_g0,cr.n_e,extra_ordinary) #signal
        
        #GV and GVD
        dk3,d2k3=1.0/GV(lambda_30,cr.sellemeir,ordinary),GVD(lambda_30,cr.sellemeir,ordinary) #pump
        dk4,d2k4=1.0/GV(lambda_40,cr.sellemeir,ordinary),GVD(lambda_40,cr.sellemeir,ordinary) #pump

    JTI=np.zeros(Ti.shape)
    if show_progress: print(JTI.shape) 

    for j in range(tau_s.size):

        if show_progress: print('\rIteration:',j,end='')

        for i in range(tau_i.size):

            S=np.zeros((w3.size,w4.size))
            for k in range(w4.size):
                
                #Gate function on the signal
                G2=G(w4[k]-Ws,1,wg0,np.sqrt(2)*sigma_g,tau_s[j])

                #Add gate dispersion
                if Ag!=0:
                    G2*=np.exp(1j*Ag*(w4[k]-Ws-wg0)**2)

                #Add phase-matching on the signal
                if add_phase_matching:

                    deltak_s=(dks*(Ws-ws0)+dkg*(w4[k]-Ws-wg0)-dk4*(w4[k]-w40)
                              +0.5*d2kg*(w4[k]-Ws-wg0)**2+0.5*d2ks*(Ws-ws0)**2
                              -0.5*d2k4*(w4[k]-w40)**2
                             )

                    phi_s=np.pi*np.exp(-1j*deltak_s*L/2)*np.sinc(1/np.pi*deltak_s*L/2)

                for l in range(w3.size):

                    #Gate function on the idler
                    G1=G(w3[l]-Wi,1,wg0,np.sqrt(2)*sigma_g,tau_i[i])

                    #Add gate dispersion
                    if Ag!=0:
                        G1*=np.exp(1j*Ag*(w3[l]-Wi-wg0)**2)

                    F_output=F_source*G1*G2
                             
                    #Add phase-matching on the idler 
                    if add_phase_matching:

                        deltak_i=(dki*(Wi-wi0)+dkg*(w3[l]-Wi-wg0)-dk3*(w3[l]-w30)
                                  +0.5*d2kg*(w3[l]-Wi-wg0)**2+0.5*d2ki*(Wi-wi0)**2
                                  -0.5*d2k3*(w3[l]-w30)**2
                                 )

                        phi_i=np.pi*np.exp(-1j*deltak_i*L/2)*np.sinc(1/np.pi*deltak_i*L/2)
                        
                        F_output*=phi_i*phi_s
            
                    #Sum over Wi and Ws
                    S[k,l]=np.abs(np.trapz(np.trapz(F_output,dx=dw),dx=dw))**2

            #Sum over w3,w4
            JTI[j,i]=np.trapz(np.trapz(S,dx=dw),dx=dw)

    return Ti,Ts,JTI

def sfg_phase_matching(W1,W3,
                       xstal='BiBO',angle=151.3,L=1e3):
    '''
    Phase-matching function for SFG.

    Parameters
    ----------
    W1: ndarray
        2D meshgrid of input frequencies
    W2: ndarray
        2D meshgrid of upconverted frequencies
    xstal: str,optional
        Type of crystal.
    angle: str, optional
        Crystal angle.
    L: float,optional
        Length of the crystal.

    Returns
    ----------
    phi:ndarray
        2D complex phase-matching function.

    '''
    
    if xstal=='BiBO':
        cr=crystal.BiBO()
        extra_ordinary=[angle,cr.ny,cr.nz] #Extra-ordinary
        ordinary=cr.nx #Ordinary 

        deltak=(cr.n_e(w2f(W1),extra_ordinary)*W1/c+cr.n_e(w2f(W3-W1),extra_ordinary)*(W3-W1)/c
                        -cr.sellemeir(w2f(W3),ordinary)*W3/c)

        phi=np.pi*np.exp(-1j*deltak*L/2)*np.sinc(1/np.pi*deltak*L/2)

    return phi


def sfg_phase_matching_withQ(W1,W3,
                             xstal='BiBO',angle=151.3,L=1e3,q=0,phiAngle=0):
    '''
    Phase-matching function for SFG with transversal dependencies.

    Parameters
    ----------
    W1: ndarray
        2D meshgrid of input frequencies
    W3: ndarray
        2D meshgrid of upconverted frequencies
    xstal: str,optional
        Type of crystal.
    angle: str, optional
        Crystal angle.
    L: float,optional
        Length of the crystal.
    q:  float, optional
        Length of transversal k-vector component
    phiAngle: float, optional
        Angle between incoming fields with frequencies W1 and W3-W1

    Returns
    ----------
    phi:ndarray
        4D complex phase-matching function.

    '''
    
    Phi=np.zeros((W1.shape[0],W1.shape[1],len(q),len(phiAngle)),dtype='complex')
    
    if xstal=='BiBO':
        cr=crystal.BiBO()
        extra_ordinary=[angle,cr.ny,cr.nz] #Extra-ordinary
        ordinary=cr.nx #Ordinary 
        
        for i in np.arange(len(q)):
            for j in np.arange(len(phiAngle)):
                q1=q[i]
                q2=q[i]
                phi=phiAngle[j]
                deltak=(np.sqrt((cr.n_e(w2f(W1),extra_ordinary)*W1/c)**2-q1**2)
                        +np.sqrt((cr.n_e(w2f(W3-W1),extra_ordinary)*(W3-W1)/c)**2-q2**2)
                    -np.sqrt((cr.sellemeir(w2f(W3),ordinary)*W3/c)**2-(q1**2+q2**2-2*q1*q2*np.cos(phi))))
                Phi[:,:,i,j]=np.pi*np.exp(-1j*deltak*L/2)*np.sinc(1/np.pi*deltak*L/2)

    return Phi

def dither_phase_matching(W1,W3,
                          angle,xstal='BiBO',L=1e3,output='sum'): 
    '''
    Phase-matching function for SFG with crystal dithering.

    Parameters
    ----------
    W1: ndarray
        2D meshgrid of input frequencies
    W2: ndarray
        2D meshgrid of upconverted frequencies
    angle: list or ndarray 
        Crystal angles.
    xstal: str,optional
        Type of crystal.
    L: float,optional
        Length of the crystal.
    output:{'sum','tensor'} str,optional
        Return sum of phase-matching functions at different angles or
        3D tensor for each crystal angle.

    Returns
    ----------
    phi:ndarray
        2D complex phase-matching function.

    '''

    Phi=np.zeros((W1.shape[0],W1.shape[1],len(angle)),dtype='complex')
    dtheta=np.diff(angle)[0]

    for i in np.arange(len(angle)):
        Phi[:,:,i]=sfg_phase_matching(W1,W3,xstal=xstal,angle=angle[i],L=L)

    if output=='tensor':
        #Return tensor with different phases.
        return Phi
    elif output=='sum':
        #Sum the abs value squared of the phase-matching functions.
        return np.trapz(np.abs(Phi)**2,dx=dtheta,axis=2)

def sfg_transfer_function(W1,W2,F_source,wg0,sigma_g,
                          Ag=0,axis=[0],add_phase_matching=False,mode='linear',xstal='BiBO',
                          angle=[151.3,146.9],L=1e3,q=0,phiAngle=0):
    '''
    Calculate sfg transfer functions. Assumes a gaussian shape for the
    gate function. Could generalize later. 

    Uses einstein summation notation to speed up calculation. 

    Parameters
    ----------
    W1: ndarray
        2D meshgrid of idler frequencies
    W2: ndarray
        2D meshgrid of signal frequencies
    F_source: ndarray
        JSA of the source
    wg0: float
        Centre frequency of the gate
    Ag: float
        Gate dispersion
    sigma_g: float
        RMS bandwidth of the gate
    add_phase_matching: bool, optional
        Add phase matching to optical gating calculation.
    L: float,optional
        Upconversion crystal length in um.

    Returns
    ----------
    X: ndarray
        Meshgrid of idler gate delays.
    Y: ndarray
        Meshgrid of singla gate delays.
    Z: ndarray
        Measured joint temporal intensity

    '''
    
    #Get center wavelengths from F_source
    w10,w20=get_moments(W1,np.abs(F_source)**2)[0],get_moments(W2.T,np.abs(F_source.T)**2)[0]

    #Define wavelengths and marginal bandwidths
    lambda_10=w2f(w10)  #idler
    lambda_20=w2f(w20)  #signal
    lambda_g0=w2f(wg0)  #gate

    #FREQUENCY SPACING
    dw=np.diff(W1[0,:])[0]

    #TEMPORAL RANGE
    T1,T2=fft_grid(W1,W2)

    #Currently defining delays tau to correspond to fft but this isn't
    #necessary.
    tau_1=T1[0,:]
    tau_2=T2[:,0]

    #UPCONVERTED FREQUENCIES
    w30,w40=w10+wg0,w20+wg0
    lambda_30,lambda_40=w2f(w30),w2f(w40)
    
    w3=(W1[0,:]-w10)+w30
    w4=(W1[0,:]-w10)+w40

    W2=(W2.T).copy()

    W3=W1.T+wg0
    W4=W2.T+wg0

    if add_phase_matching:
        #Calculate crystal axes parameters

        if 1 in axis:
            if mode=='linear':
                extra_ordinary=[angle[0],cr.ny,cr.nz] #Extra-ordinary
                ordinary=cr.nx #Ordinary 

                dk1,d2k1=1.0/GV(lambda_10,cr.n_e,extra_ordinary),GVD(lambda_10,cr.n_e,extra_ordinary) #idler
                dk3,d2k3=1.0/GV(lambda_30,cr.sellemeir,ordinary),GVD(lambda_30,cr.sellemeir,ordinary) #pump
                dkg,d2kg=1.0/GV(lambda_g0,cr.n_e,extra_ordinary),GVD(lambda_g0,cr.n_e,extra_ordinary) #signal

                #print(c*dk1,c*dkg,c*dk3)
                #print(d2k1,d2kg,d2k3)

                deltak_1=(dk1*(W1-w10)+dkg*(W3-W1-wg0)-dk3*(W3-w30)
                          # +0.5*d2kg*(W3-W1-wg0)**2+0.5*d2k1*(W1-w10)**2
                          #    -0.5*d2k3*(W3-w30)**2
                             )
                phi1=np.pi*np.exp(-1j*deltak_1*L/2)*np.sinc(1/np.pi*deltak_1*L/2)

            elif mode=='full':
                phi1=sfg_phase_matching(W1,W3,angle=angle[0],L=L)
                
            elif mode=='full_withQ':
                phi1=sfg_phase_matching_withQ(W1,W3,angle=angle[0],L=L,q=q,phiAngle=phiAngle)

            elif mode=='dither':
                phi1=dither_phase_matching(W1,W3,angle[0],L=L,output='tensor')

        if 0 in axis:
            if mode=='linear':
                extra_ordinary=[angle[1],cr.ny,cr.nz] #Extra-ordinary
                ordinary=cr.nx #Ordinary 

                dk2,d2k2=1.0/GV(lambda_20,cr.n_e,extra_ordinary),GVD(lambda_20,cr.n_e,extra_ordinary) 
                dk4,d2k4=1.0/GV(lambda_40,cr.sellemeir,ordinary),GVD(lambda_40,cr.sellemeir,ordinary) 
                dkg,d2kg=1.0/GV(lambda_g0,cr.n_e,extra_ordinary),GVD(lambda_g0,cr.n_e,extra_ordinary) #signal

                #print(c*dk2,c*dkg,c*dk4)
                #print(d2k2,d2kg,d2k4)

                deltak_2=(dk2*(W2-w20)+dkg*(W4-W2-wg0)-dk4*(W4-w40)
                          #    +0.5*d2kg*(W4-W2-wg0)**2+0.5*d2k2*(W2-w20)**2
                          #    -0.5*d2k4*(W4-w40)**2
                             )
                phi2=np.pi*np.exp(-1j*deltak_2*L/2)*np.sinc(1/np.pi*deltak_2*L/2)    

            elif mode=='full':
                phi2=sfg_phase_matching(W2,W4,angle=angle[1],L=L)
                
            elif mode=='full_withQ':
                phi2=sfg_phase_matching_withQ(W2,W4,angle=angle[1],L=L,q=q,phiAngle=phiAngle)

            elif mode=='dither':
                phi2=dither_phase_matching(W2,W4,angle[1],L=L,output='tensor')
    #TRANSFER FUNCTION FOR IDLER SIDE
    if 1 in axis:

        #Calculate gate functions (eventually replace with measured gate?)
        W1p,W3p,Tau1=np.meshgrid(W1[0,:],W3[:,0],tau_1)
        G1=G(W3p-W1p,1,wg0,sigma_g,Tau1)

        #Add gate dispersion
        if Ag!=0:
            G1*=np.exp(1j*Ag*(W3p-W1p-wg0)**2)
        
        #Add phase-matching
        if add_phase_matching:
            if mode=='dither':
                T1=np.einsum(G1,[3,0,2],np.conjugate(G1),[3,1,2],
                         phi1,[3,0,4],np.conjugate(phi1),[3,1,4],[0,1,2],optimize=True)
                
            if mode=='full_withQ':
                T1=np.einsum(G1,[3,0,2],np.conjugate(G1),[3,1,2],
                         phi1,[3,0,4,5],np.conjugate(phi1),[3,1,4,5],[0,1,2],optimize=True)
            else:
            #Calculate tensor transfer function with phase-matching
                T1=np.einsum(G1,[3,0,2],np.conjugate(G1),[3,1,2],
                         phi1,[3,0],np.conjugate(phi1),[3,1],[0,1,2],optimize=True)
        else:
            #Calculate tensor transfer function without phase-matching
            T1=np.einsum(G1,[3,0,2],np.conjugate(G1),[3,1,2],[0,1,2],optimize=True)

    #TRANSFER FUNCTION FOR SIGNAL SIDE
    if 0 in axis:
        #Calculate gate function
        W2p,W4p,Tau2=np.meshgrid(W2[0,:],W4[:,0],tau_2)
        G2=G(W4p-W2p,1,wg0,sigma_g,Tau2)
        if Ag!=0:
            G2*=np.exp(1j*Ag*(W4p-W2p-wg0)**2)

        if add_phase_matching:
            if mode=='dither':
                T2=np.einsum(G2,[3,0,2],np.conjugate(G2),[3,1,2],
                         phi2,[3,0,4],np.conjugate(phi2),[3,1,4],[0,1,2],optimize=True)
                
            if mode=='full_withQ':
                T2=np.einsum(G2,[3,0,2],np.conjugate(G2),[3,1,2],
                         phi2,[3,0,4,5],np.conjugate(phi2),[3,1,4,5],[0,1,2],optimize=True)

            else:
            #Calculate tensor transfer function with phase-matching
                T2=np.einsum(G2,[3,0,2],np.conjugate(G2),[3,1,2],
                         phi2,[3,0],np.conjugate(phi2),[3,1],[0,1,2],optimize=True)
        else:
            #Calculate tensor transfer function without phase-matching
            T2=np.einsum(G2,[3,0,2],np.conjugate(G2),[3,1,2],[0,1,2],optimize=True)
    
    if (1 in axis) and (0 in axis):
        return T1,T2

    elif 1 in axis:
        return T1

    elif 0 in axis:
        return T2

    #UPCONVERT BOTH SIDES
def model_photon_sfg(W1,W2,F_source,T,
                     axis=[1]):
    '''
    Model the sfg using the transfer functions calculated in
    sfg_transfer_function.

    Uses einstein summation notation to speed up calculation. 

    Parameters
    ----------
    W1: ndarray
        2D meshgrid of idler frequencies
    W2: ndarray
        2D meshgrid of signal frequencies
    F_source: ndarray
        JSA of the source
    T: ndarray
        Transfer function for sfg
    axis:list 
        Specifies which axes to apply the transfer function to.  

    Returns
    ----------
    X: ndarray
        Meshgrid on idler side.
    Y: ndarray
        Meshgrid of signal side. 
    Z: ndarray
        Intensity at X,Y coordinates.

    '''
    T1,T2=fft_grid(W1,W2)

    tau_1,tau_2=T1[0,:],T2[:,0]

    if (1 in axis) and (0 in axis):
        T1,T2=T
        X,Y=np.meshgrid(tau_1,tau_2)
        Z=np.abs(np.einsum(T1,[0,1,2],T2,[3,4,5],F_source,[3,0],np.conjugate(F_source),[4,1],[5,2],optimize=True))

    #UPCONVERT IDLER
    elif 1 in axis:
        T1=T
        X,Y=np.meshgrid(tau_1,W2[:,0])
        Z=np.abs(np.einsum(T1,[0,1,2],F_source,[3,0],np.conjugate(F_source),[3,1],[3,2],optimize=True))

    #UPCONVERT SIGNAL 
    elif 0 in axis:
        T2=T
        X,Y=np.meshgrid(W1[0,:],tau_2)
        Z=np.abs(np.einsum(T2,[0,1,2],F_source,[0,3],np.conjugate(F_source),[1,3],[2,3],optimize=True))
        
    return X,Y,Z

def fast_double_upconversion(Wi,Ws,F_source,wg0,sigma_g,
                             Ag=0,L=1e3,show_progress=True):
    '''
    Calculate expected joint temporal intensity by optically gating two
    SPDC photons using fftconvolve. Assumes a gaussian shape for the gate
    function. For the moment, can't include phase-matching. 

    Parameters
    ----------
    Wi: ndarray
        2D meshgrid of idler frequencies
    Ws: ndarray
        2D meshgrid of signal frequencies
    F_source: ndarray
        JSA of the source
    wg0: float
        Centre frequency of the gate
    sigma_g: float
        RMS bandwidth of the gate
    Ag: float
        gate dispersion
    add_phase_matching: bool, optional
        Add phase matching to optical gating calculation.
    L: float,optional
        Upconversion crystal length in um.

    Returns
    ----------
    Ti: ndarray
        Meshgrid of idler gate delays.
    Ts: ndarray
        Meshgrid of singla gate delays.
    JTI: ndarray
        Measured joint temporal intensity

    '''

    wi0,ws0=get_moments(Wi,np.abs(F_source)**2)[0],get_moments(Ws.T,np.abs(F_source.T)**2)[0]

    #Define wavelengths and marginal bandwidths
    lambda_i0=w2f(wi0)  #idler
    lambda_s0=w2f(ws0)  #signal
    lambda_g0=w2f(wg0)  #gate

    #FREQUENCY SPACING
    dw=np.diff(Wi[0,:])[0]

    #TEMPORAL RANGE
    Ti,Ts=fft_grid(Wi,Ws)
    tau_i=Ti[0,:]
    tau_s=Ts[:,0]

    #UPCONVERTED FREQUENCIES
    w30,w40=wi0+wg0,ws0+wg0
    lambda_30,lambda_40=w2f(w30),w2f(w40)
    
    w3=(Wi[0,:]-wi0)+w30
    w4=(Wi[0,:]-wi0)+w40

    JTI=np.zeros(Ti.shape)
    if show_progress: print(JTI.shape) 

    for j in range(tau_s.size):

        if show_progress: print('\rIteration:',j,end='')

        for i in range(tau_i.size):

            S=np.zeros((w3.size,w4.size))

            F_output=F_source

            #Gate functions on the signal and idler
            G1=G(Wi-wi0+wg0,1,wg0,np.sqrt(2)*sigma_g,tau_i[i])
            G2=G(Ws-ws0+wg0,1,wg0,np.sqrt(2)*sigma_g,tau_s[j])

            #Add gate dispersion:
            if Ag!=0:
                G1=G1*np.exp(1j*Ag*(Wi-wi0)**2)
                G2=G2*np.exp(1j*Ag*(Ws-ws0)**2)

            
            #Convolve W1 with w3 and W2 with w4
            S=np.abs(fftconvolve(F_output,G1*G2))**2

            #Sum over w3, w4
            JTI[j,i]=np.trapz(np.trapz(S,dx=dw),dx=dw)

    return Ti,Ts,JTI

def model_gaussian_sfg(w10,sigma_1,w20,sigma_2,
                       A=0,L=1e3,dz=1,dt=10):
    '''
    Calculate classical upconversion signal field (w3=w1+w2) for optically
    gating a photon including phase-matching. Assumes a gaussian shape for
    the photon and the gate.  
    
    See Weiner, Ultrafast Optics, page 240.

    Parameters
    ----------
    w10: ndarray
        Centre frequency of the photon
    sigma_1: float
        RMS bandwidth of the photon 
    w20: ndarray
        Centre frequency of the gate
    sigma_2: float
        RMS bandwidth of the gate
    A: float
        Photon dispersion
    L: float,optional
        Upconversion crystal length in um.

    Returns
    ----------
    tau: ndarray
        Gate delays.
    I3: ndarray
        Integrated intensity for each gate delay.
    '''
   
    extra_ordinary=[angle,cr.ny,cr.nz] #Extra-ordinary
    ordinary=cr.nx #Ordinary 

    w30=w10+w20
    lambda_10,lambda_20,lambda_30=w2f(w10),w2f(w20),w2f(w30)
        
    dk1=1.0/GV(lambda_10,cr.n_e,extra_ordinary) #idler
    dk2=1.0/GV(lambda_20,cr.n_e,extra_ordinary) #signal
        
    #GV and GVD
    dk3=1.0/GV(lambda_30,cr.sellemeir,ordinary)#pump

    eta13=(dk1-dk3)
    eta23=(dk2-dk3)
    
    twidth=np.sqrt(0.25/sigma_2**2+0.25/sigma_1**2+4*A**2*sigma_1**2)
    trange=2*2.35*twidth
    
    T,Z=np.meshgrid(np.arange(-trange,trange,dt),np.arange(0,L,dz))
    
    W,Ws=fft_grid(T,T)
    
    tau=np.linspace(-trange,trange,80)
    I3=np.zeros(tau.size)

    #Field amplitude 1
    a1=gauss(T-eta13*Z,1,0,np.sqrt(2)*1/(2*sigma_1),0)
    #Add dispersion
    aw1=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(a1,axes=[1]),axes=[1]),axes=[1])*np.exp(1j*A*W**2)
    a1=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(aw1,axes=[1]),axes=[1]),axes=[1])

    for i in range(tau.size):
        print('\rIteration',i+1,'of',tau.size,end='')
        #Field amplitude 2
        a2=gauss(T-eta23*Z-tau[i],1,0,np.sqrt(2)*1/(2*sigma_2),0)
        #Upconverted intensity
        I3[i]=np.trapz(np.abs(np.trapz(a1*a2,dx=dz,axis=0))**2,dx=dt)
    
    return tau,I3

def model_sfg_spectrogram(W1,W1p,F_mixed,wg0,sigma_g,
                          Ag=0,add_phase_matching=False,mode='linear',xstal='BiBO',
                          angle=[151.3,146.9],L=1e3,q=0,phiAngle=0):
    '''
    Model the sfg using the transfer functions calculated in
    sfg_transfer_function.

    Uses einstein summation notation to speed up calculation. 

    Parameters
    ----------
    W1: ndarray
        2D meshgrid of frequencies
    W1p: ndarray
        2D meshgrid of frequencies
    F_mixed: ndarray
        JSA of the source
    wg0: float
        Centre frequency of the gate
    Ag: float
        Gate dispersion
    sigma_g: float
        RMS bandwidth of the gate
    add_phase_matching: bool, optional
        Add phase matching to optical gating calculation.
    L: float,optional
        Upconversion crystal length in um.
    Returns
    ----------
    X: ndarray
        Meshgrid on delays tau.
    Y: ndarray
        Meshgrid of upconverted frequencies. 
    Z: ndarray
        Spectrogram at X,Y coordinates.

   '''
        
    #Get center wavelengths from F_source
    w10=get_moments(W1[0,:],np.real(np.diag(F_mixed)))[0]

    #Define wavelengths and marginal bandwidths
    lambda_10=w2f(w10)  #idler
    lambda_g0=w2f(wg0)  #gate

    #FREQUENCY SPACING
    dw=np.diff(W1[0,:])[0]

    #TEMPORAL RANGE
    T1,T1p=fft_grid(W1,W1p)

    #Currently defining delays tau to correspond to fft but this isn't
    #necessary.
    tau_1=T1[0,:]

    #UPCONVERTED FREQUENCIES
    w30=w10+wg0
    lambda_30=w2f(w30)
    
    w3=(W1[0,:]-w10)+w30
    W3=W1.T+wg0

    if add_phase_matching:
        #Calculate crystal axes parameters

        if mode=='linear':
            extra_ordinary=[angle[0],cr.ny,cr.nz] #Extra-ordinary
            ordinary=cr.nx #Ordinary 

            dk1,d2k1=1.0/GV(lambda_10,cr.n_e,extra_ordinary),GVD(lambda_10,cr.n_e,extra_ordinary) #idler
            dk3,d2k3=1.0/GV(lambda_30,cr.sellemeir,ordinary),GVD(lambda_30,cr.sellemeir,ordinary) #pump
            dkg,d2kg=1.0/GV(lambda_g0,cr.n_e,extra_ordinary),GVD(lambda_g0,cr.n_e,extra_ordinary) #signal

            #print(c*dk1,c*dkg,c*dk3)
            #print(d2k1,d2kg,d2k3)

            deltak_1=(dk1*(W1-w10)+dkg*(W3-W1-wg0)-dk3*(W3-w30)
                      # +0.5*d2kg*(W3-W1-wg0)**2+0.5*d2k1*(W1-w10)**2
                      #    -0.5*d2k3*(W3-w30)**2
                         )
            phi1=np.pi*np.exp(-1j*deltak_1*L/2)*np.sinc(1/np.pi*deltak_1*L/2)

                          
    #TRANSFER FUNCTION FOR IDLER SIDE

    #Calculate gate functions (eventually replace with measured gate?)
    W1p,W3p,Tau1=np.meshgrid(W1[0,:],W3[:,0],tau_1)
    G1=G(W3p-W1p,1,wg0,sigma_g,Tau1)

    #Add gate dispersion
    if Ag!=0:
        G1*=np.exp(1j*Ag*(W3p-W1p-wg0)**2)

    #Add phase-matching
    if add_phase_matching:

        #Calculate tensor transfer function with phase-matching
            S=np.einsum(G1,[3,0,2],np.conjugate(G1),[3,1,2],
                     phi1,[3,0],np.conjugate(phi1),[3,1],F_mixed,[0,1],[3,2],optimize=True)
    else:
        #Calculate tensor transfer function without phase-matching
        S=np.einsum(G1,[3,0,2],np.conjugate(G1),[3,1,2],F_mixed,[0,1],[3,2],optimize=True)

    return T1,W3,np.abs(S)





        


