'''
###############################################################################
IO MODULE
###############################################################################

Routines in this module:

Input
-----
> get_data(folder,file_search_term,file_number,columns)
> read_spectrogram(folder,file_search_term,file_number)
> load_spectrogram(folder,file_search_term,file_number)
> load_2d_data(folder,search_term,file_number,columns)
> load_gs_data(folder,search_term,file_number)
> create_tf_data(frequency_data)

Output
------
> write_FROG_spectrogram(folder)

Plot/Print
----------
> plot_spectrogram_data(T,W,S)
> plot_2D_Gaussian(X,Y,Z)
> plot_tf_data(gs_data)
> print_tf_fit_data(gs_data)
> plot_gaussian(X,Y,Z)

'''

import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os import path
from re import search

from ..analytics import *

from .binner import *
from .postProcessing import *

c=0.299792458 #Speed of light in um/fs or mm/ps

'''
-------------------------------------------------------------------------------
Input
-------------------------------------------------------------------------------
'''
def get_data(folder,file_search_term,file_number,columns,
             skiprows=21):
    '''
    Load columns in a text file. Can specify 2, 3, or arbitrary number of
    columns.

    Parameters
    ----------
    folder: str
        Folder directory.

    file_search_term: str 
        String in the name of the file to find file.

    file_number: int 
        Specify which file number in the list of files with
        the string from file_search_term to use.

    columns: list of int 
        A list of integers to specify which columns in the file to read.

    skiprows: int, optional
        Number of rows to skip when reading file.

    Returns
    ----------
    out:
        Data read from the text file

    '''

    files=[f for f in listdir(folder) if search(file_search_term,f)]

    #if len(files)>1:print 'Too many files!'
    #Set to import only one file. Could add possibility to immport
    #multiple files.
    files=files[file_number]

    if len(columns)==2:
        data=np.loadtxt(folder+files,skiprows=skiprows) 
        x,y=data[:,columns[0]], data[:,columns[1]]   
        return x,y

    elif len(columns)==3:
        data=np.loadtxt(folder+files,skiprows=skiprows) 
        x,y,z=data[:,columns[0]], data[:,columns[1]], data[:,columns[2]]  
        return x,y,z

    else: 
        data=np.loadtxt(folder+files,skiprows=skiprows) 
        return data[:,columns]
    
def read_spectrogram(folder,file_search_term,file_number,
                     skiprows=8):
    '''
    Read spectrogram file output by labview VI.

    Parameters
    ----------
    folder: str
        folder directory 

    file_search_term: str 
        string in file to specifiy file to use 

    file_number: int 
        if more than one file with same string, specify which
        file_number in list to use

    skiprows: int, optional
        Number of rows to skip when reading file.

    Returns
    ----------
    out: tuple (1D array, 1D array, 2D array)
        Data read from spectrogram file. First variable is a 1D array of
        delays, second variable is a 1D array of wavelengths, and third
        variable is a 2D array of the measured spectrogram intensity or
        counts.
    '''

    files=[f for f in listdir(folder) if search(file_search_term,f)]

    #if len(files)>1:print 'Too many files!'
    files=files[file_number]

    data=np.loadtxt(folder+files,skiprows=skiprows) 

    x,y,z=data[0,1:], data[1:,0], data[1:,1:]  

    return x,y,np.array(z)

def load_spectrogram(folder,file_search_term,file_number,
                     background=False,axes=['time','wavelength'],limits=[],skiprows=8):
    '''
    Read spectrogram file output by labview VI.

    Parameters
    ----------
    folder: str
        folder directory 

    file_search_term: str 
        string in file to specifiy file to use 

    file_number: int 
        if more than one file with same string, specify which
        file_number in list to use

    bacground: bool, optional
        Background subtract another file.

    axes: [{'None','time'},{'wavelength','frequency'}] list of str, optional
        Units of x and y axes.

    limits: [xmin,xmax,ymin,ymax], optional
        x and y axis limits to read

    skiprows: int, optional
        Number of rows to skip when reading file.

    Returns
    ----------
    out: tuple (2D array, 2D array, 2D array)
        X,Y meshgrid and Z Data read from spectrogram file. First variable is a 1D array of
        delays, second variable is a 1D array of wavelengths, and third
        variable is a 2D array of the measured spectrogram intensity or
        counts.
    '''


    x,y,Z=read_spectrogram(folder,file_search_term,file_number,skiprows=skiprows)

    if background:
        x,y,Zbg=read_spectrogram(folder,file_search_term,file_number+1,skiprows=skiprows)
        Z=Z-Zbg
        Z[Z<0]=0

    x0=np.sum(Z.sum(0)*x)/Z.sum()
    y0=np.sum(Z.sum(1)*y)/Z.sum()

    if axes[0]=='time':
        #The gate starts ahead of the photons which corresponds to
        #positive delay.
        x=-(x-x0)*1e3/c

    if axes[1]=='frequency':
        y=2*np.pi*c/(y*1e-3)

    if len(limits)==4:
        xmin,xmax,ymin,ymax=limits

        xroi=np.where((x<=xmax) & (x>=xmin))
        x=x[xroi]
        Z=Z[:,xroi[0]]

        yroi=np.where((y<=ymax) & (y>=ymin))
        y=y[yroi]
        Z=Z[yroi[0],:]

    X,Y=np.meshgrid(x,y) 

    return X,Y,Z

def load_2d_data(folder,search_term,file_number,columns,
                 axes=[],origin=[],grid='data',method='nearest',skiprows=21):
    '''
    Load frequency or time data as a 2-D grid

    Parameters
    ----------
    folder: str
        Folder directory.
    search_term: str 
        String in the name of the file to find file.
    file_number: int 
        Specify which file number in the list of files with
        the string from file_search_term to use.
    columns: list of int 
        A list of integers to specify which columns in the file to read.
    axes:list, {'frequency','wavelength','time','motor'}
        Specify what units to use for both axes, e.g. ['frequency','time'] 
    grid:
        See grid2d_data.
    method:str,optional
        Interpolation method for grid. See grid2d_data.
    skiprows: int, optional
        Number of rows to skip when reading file.

    Returns
    ----------
    Data read from the files
    out:tuple
        (x,y,z) 1D arrays of coordinates.
        or 
        (X,Y,Z)a meshgrid of gridded coordinates.

    '''

    x,y,z=get_data(folder,search_term,file_number,columns,skiprows=skiprows)

    pfit=fit_gaussian((x,y,z)).x

    if origin:
        x0,y0=origin
    else:
        A,x0,y0,xsigma,ysigma,rho,k=pfit

    if axes[0]=='frequency':
        x=3.17708*x+787.111 #Idler spectrometer
        x=2*np.pi*c/x*1e3
    elif axes[0]=='wavelength':
        x=3.17708*x+787.111 #Idler spectrometer
    elif axes[0]=='time':
        x=-(x-x0)/c
    elif axes=='motor':
        x,y=x,y
    else:
        print('Please specify time, frequency, wavelength, or motor axes.')

    if axes[1]=='frequency':
        y=-3.49495*y+763.105 #Signal spectrometer
        y=2*np.pi*c/y*1e3
    elif axes[1]=='wavelength':
        y=-3.49495*y+763.105 #Signal spectrometer
    elif axes[1]=='time':
        y=-(y-y0)/c
    else: 
        print('Please specify time, frequency, or wavelength axes.')

    if grid:
        X,Y,Z=grid2d_data(x,y,z,grid=grid,method=method)
        if np.isnan(Z).any():
            Z[np.isnan(Z)]=0
        return X,Y,Z
    else:
        return x,y,z

def load_gs_data(folder,search_term,file_number,
                 grid='data',grid_size=0,fft_time=False,pad=False,
                 background_subtraction=False,d=[8,8,8,8],
                 deconvolve=False,alpha=[0.1,0.1,0.1,0.1],low_pass=[0,0,0,0],epsilon=[0,0,0,0],
                 instrument_resolution=[0.0001,0.0001,120,120]):
    '''
    Load and process data for the modified GS Algorithm which requires all
    four plots. Files must be in the same folder. 

    Parameters
    ----------
    folder: str
        Folder directory.

    search_term: str 
        String in the name of the file to find file.

    file_number: int 
        Specify which file number in the list of files with
        the string from file_search_term to use.

    ...to be continued

    Returns
    ----------
    out:
        Data read from the files

    '''

    wi,ws,iww=load_2d_data(folder,search_term,file_number[0],[1,3,4],
                           ['frequency','frequency'],grid=False)

    ti1,ws1,itw=load_2d_data(folder,search_term,file_number[1],[0,1,2],
                             ['time','frequency'],grid=False)
    #roi=np.where(itw<=220)
    #ti1,ws1,itw=ti1[roi],ws1[roi],itw[roi]

    wi2,ts2,iwt=load_2d_data(folder,search_term,file_number[2],[0,1,2],
                             ['frequency','time'],grid=False)

    ti,ts,itt=load_2d_data(folder,search_term,file_number[3],[0,1,2],['time','time'],grid=False)

    wi0,ws0=fit_gaussian((wi,ws,iww)).x[1:3]
   

    if grid=='data':
        Wi,Ws,Iww=grid2d_data(wi,ws,iww)
        Ti1,Ws1,Itw=grid2d_data(ti1,ws1,itw)
        Wi2,Ts2,Iwt=grid2d_data(wi2,ts2,iwt)
        Ti,Ts,Itt=grid2d_data(ti,ts,itt)


    elif grid=='double_time':
        #Grid based on specified plot
        Ti,Ts,Itt=grid2d_data(ti,ts,itt,grid='double_time',method='cubic')
        Wi,Ws,Iww=grid2d_data(wi,ws,iww,grid='data',method='linear')
        Ti1,Ws1,Itw=grid2d_data(ti1,ws1,itw,grid='data',method='nearest')
        Wi2,Ts2,Iwt=grid2d_data(wi2,ts2,iwt,grid='data',method='nearest')
        Iww[np.isnan(Iww)]=0
        Iww[np.where(Iww<0)]=0
        Itt[np.isnan(Itt)]=0
        Itt[np.where(Itt<0)]=0

    else:
        Ti,Ts,Itt=grid2d_data(ti,ts,itt,grid='data',method='nearest')
        Wi,Ws,Iww=grid2d_data(wi,ws,iww,grid='data',method='linear')
        Ti1,Ws1,Itw=grid2d_data(ti1,ws1,itw,grid='data',method='cubic')
        Wi2,Ts2,Iwt=grid2d_data(wi2,ts2,iwt,grid='data',method='cubic')

        Iww[np.isnan(Iww)]=0
        Iww[np.where(Iww<0)]=0

        Iwt[np.isnan(Iwt)]=0
        Iwt[np.where(Iwt<0)]=0

        Itw[np.isnan(Itw)]=0
        Itw[np.where(Itw<0)]=0

        #TIME HACK  
        Ti*=1e3
        Ts*=1e3
        Ti1*=1e3
        Ts2*=1e3
    
    if background_subtraction:
        Iww=corner_subtract(Iww,d[0])
        Itw=corner_subtract(Itw,d[1])
        Iwt=corner_subtract(Iwt,d[2])
        Itt=corner_subtract(Itt,d[3])
    
    if pad:
        #PAD all to the next power of 2
        Wi,Ws,Iww=padnextpow2(Wi,Ws,Iww,grid_size=grid_size)
        Ti1,Ws1,Itw=padnextpow2(Ti1,Ws1,Itw,grid_size=grid_size)
        Wi2,Ts2,Iwt=padnextpow2(Wi2,Ts2,Iwt,grid_size=grid_size)
        Ti,Ts,Itt=padnextpow2(Ti,Ts,Itt,grid_size=grid_size)

    elif fft_time:
            if isinstance(grid,tuple):
                #Grid based on specified plot
                Ti,Ts,Itt=grid2d_data(Ti.flatten(),Ts.flatten(),Itt.flatten(),grid=grid,
                                      method='cubic')
            else:
                #If grid size if specified
                Ti,Ts,Itt=padnextpow2(Ti,Ts,Itt,grid_size=grid_size)

            #elif grid=='double_time':
            #    #Grid based on specified plot
            #    Ti,Ts,Itt=grid2d_data(Ti.flatten(),Ts.flatten(),Itt.flatten(),grid='data',method='linear')
            #    #Ti,Ts,Itt=grid2d_data(ti,ts,itt,grid='double_time',method='linear')
            #    Ti,Ts,Itt=padnextpow2(Ti,Ts,Itt,grid_size=grid_size)

            Wig,Wsg=fft_grid(Ti,Ts)
            #Wig=Wig/1000.0+wi0
            #Wsg=Wsg/1000.0+ws0
            Wig=Wig+wi0
            Wsg=Wsg+ws0

            Wi,Ws,Iww=grid2d_data(Wi.flatten(),Ws.flatten(),Iww.flatten(),
                                            grid=(Wig,Wsg),method='cubic')
            Ti1,Ws1,Itw=grid2d_data(Ti1.flatten(),Ws1.flatten(),Itw.flatten(),
                                            grid=(Ti,Wsg),method='cubic')
            Wi2,Ts2,Iwt=grid2d_data(Wi2.flatten(),Ts2.flatten(),Iwt.flatten(),
                                                 grid=(Wig,Ts),method='cubic')

           # Wi,Ws,Iww=grid2d_data(wi,ws,iww,grid=(Wig,Wsg),method='linear')
           # Ti1,Ws1,Itw=grid2d_data(ti1,ws1,itw,grid=(Ti,Wsg),method='linear')
           # Wi2,Ts2,Iwt=grid2d_data(wi2,ts2,iwt,grid=(Wig,Ts),method='linear')

    Iww[np.isnan(Iww)]=0
    Iww[np.where(Iww<0)]=0

    Itw[np.isnan(Itw)]=0
    Itw[np.where(Itw<0)]=0

    Iwt[np.isnan(Iwt)]=0
    Iwt[np.where(Iwt<0)]=0

    Itt[np.isnan(Itt)]=0
    Itt[np.where(Itt<0)]=0

    if deconvolve:

        spect_1,spect_2,gate_1,gate_2=instrument_resolution
        Pii=wi0**2/(2*np.pi*c)*spect_1
        Pis=ws0**2/(2*np.pi*c)*spect_2

        gww=gauss2d([Wi,Ws],1.0,wi0,ws0,Pii,Pis,0,0)
        gtw=gauss2d([Ti1,Ws1],1.0,0,ws0,gate_1,Pis,0,0)
        gwt=gauss2d([Wi2,Ts2],1.0,wi0,0,Pii,gate_2,0,0)
        gtt=gauss2d([Ti,Ts],1.0,0,0,gate_1,gate_2,0,0)

        Idww=np.abs(fftdeconvolve(Iww,gww,alpha=alpha[0],low_pass=low_pass[0],epsilon=epsilon[0]))
        Idtw=np.abs(fftdeconvolve(Itw,gtw,alpha=alpha[1],low_pass=low_pass[1],epsilon=epsilon[1]))
        Idwt=np.abs(fftdeconvolve(Iwt,gwt,alpha=alpha[2],low_pass=low_pass[2],epsilon=epsilon[2]))
        Idtt=np.abs(fftdeconvolve(Itt,gtt,alpha=alpha[3],low_pass=low_pass[3],epsilon=epsilon[3]))
        
        Iww,Itw,Iwt,Itt=Idww,Idtw,Idwt,Idtt

    return Wi,Ws,Iww,Ti1,Ws1,Itw,Wi2,Ts2,Iwt,Ti,Ts,Itt

def create_tf_data(frequency_data,
                   add_poisson_noise=True,Nphotons=300):
    '''
    Create time frequency data for modified GS algorithm from .

    Parameters
    ----------
    frequency_data:tuple
        Wi: ndarray 
            2D meshgrid of Idler frequencies
        Ws: ndarray 
            2D array of Signal frequencies.     
        F: ndarray 
            2D array of Frequency-Frequency amplitudes.     
    add_poisson_noise:bool
        Add Poisson noise to counts
    Nphotons:float
        Average number of photons for Poisson statistics.

    Returns
    ---------- 
    Wi: ndarray 
        2D meshgrid of Idler frequencies
    Ws: ndarray 
        2D array of Signal frequencies.     
    Ti: ndarray 
        2D array of Idler time     
    Ts: ndarray 
        2D array of Signal time     
    Iww: ndarray 
        2D array of Frequency-Frequency count data.     
    Itw: ndarray 
        2D array of Time-Frequency data.     
    Iwt: ndarray 
        2D array of Frequency-Time data.     
    Itt: ndarray 
        2D array of Time-Time data.     
    '''
    Wi,Ws,F=frequency_data
    _,_,fwt=fft_state_1D(Wi,Ws,F,axis=0)
    _,_,ftw=fft_state_1D(Wi,Ws,F,axis=1)
    Ti,Ts,ftt=fft_state_2D(Wi,Ws,F)

    if add_poisson_noise:
        Iww=np.double(np.random.poisson(Nphotons*np.abs(F)**2))
        Iwt=np.double(np.random.poisson(Nphotons*(np.abs(fwt)/np.max(np.abs(fwt)))**2))
        Itw=np.double(np.random.poisson(Nphotons*(np.abs(ftw)/np.max(np.abs(ftw)))**2))
        Itt=np.double(np.random.poisson(Nphotons*(np.abs(ftt)/np.max(np.abs(ftt)))**2))
    else:
        Iww=np.double(np.abs(F)**2)
        Iwt=np.double(np.abs(fwt)**2/np.max(np.abs(fwt)**2))
        Itw=np.double(np.abs(ftw)**2/np.max(np.abs(ftw)**2))
        Itt=np.double(np.abs(ftt)**2/np.max(np.abs(ftt)**2))
        
    return Wi,Ws,Ti,Ts,Iww,Itw,Iwt,Itt
    
'''
-------------------------------------------------------------------------------
Output
-------------------------------------------------------------------------------
'''
def write_FROG_spectrogram(folder, spectrogramFileName='Laser_spectrogram',searchTerm='FROG',delayRange=[],skiprows=10):
    '''
    Read FROG spectra from AC measurement and write Laser_spectrogram.txt

    Parameters
    ----------
    folder: str
        Specify folder path to AC measurement data
    spectrogramFile: str, optional
        Specify output file name
    searchTerm: str, optional
        Specify string in file name to find file
    delayRange: [delayMin,delayMax,delayStepSize]
        Specify the absolute position range of AC measurement in cm
    skiprows: int, optional
        Number of rows to skip when reading file

    Returns
    ----------
    2D array saved as ASCI file to directory (path: folder), if file with spectrogramFileName doesn't already exist

    '''
    
    if path.isfile(folder+spectrogramFileName+'.txt') == False:
        fileNames=[f for f in listdir(folder) if search(searchTerm,f)]
        n=len(fileNames)

        w,_=get_data(folder,searchTerm,0,[0,1],skiprows=skiprows)
    
        x=np.arange(delayRange[0],delayRange[1],delayRange[2])

        S=np.zeros((w.size,n))
        for i in range(n):
            _,S[:,i]=get_data(folder,searchTerm,i,[0,1],skiprows=skiprows)

        x0=np.sum(x*S.sum(0))/np.sum(S)
        t=2*(x-x0)*1000/c

        T,W=np.meshgrid(t,w)

        A=np.hstack((w.reshape(w.size,1),S))
        A=np.vstack((np.hstack(([999],t)),A))

        np.savetxt(folder+spectrogramFileName+'.txt',A,delimiter='\t')
    


'''
-------------------------------------------------------------------------------
Plot
-------------------------------------------------------------------------------
'''
def plot_spectrogram_data(T,W,S,
                          title='Measured'):
    '''
    Plot Spectrogram data.

    Parameters
    ----------
    T: 2D array,
        Time x-axis meshgrid array
    W: 2D array,
        Frequency (or wavelength) y-axis meshgrid array 
    S: 2D array,
        Spectrogram array

    Returns
    ----------
    (T,W,S) pcolormesh plot of spectogram.

    '''

    fig,ax=plt.subplots(1,3,figsize=(14,4))
    ax[0].set_title(title)
    S1=ax[0].pcolormesh(T,W,S)
    ax[0].set_xlabel('Time (fs)')
    ax[0].set_ylabel(r'Frequency (fs$^{-1}$)')
    fig.colorbar(S1,ax=ax[0])

    ax[1].set_title('Frequency marginal')
    ax[1].plot(W[:,0],S.sum(1))
    ax[1].set_xlabel(r'Frequency (fs$^{-1}$)')

    ax[2].set_title('Time marginal')
    ax[2].plot(T[0,:],S.sum(0),'C1')
    ax[2].set_xlabel('Time (fs)')

    plt.tight_layout()
    plt.show()
    
def plot_2D_Gaussian(X,Y,Z,
                     fit_plot=True,marginals=False,text_box=True,
                     print_fit_parameters=False,gate_resolution=[[0.120,0.003],[0.120,0.003]]):
    '''
    Make a 2D contour plot of measured data. Fit to gaussian if desired.

    Parameters
    ----------
    X: 2D array,
        X-axis meshgrid array
    Y: 2D array,
        Y-axis meshgrid array 
    Z: 2D array,
        Z axis array.
    fit_plot:bool
        Fit array to 2D Gaussian
    marginal:bool
        Plot marginals.
    text_box:bool
        Show text box of fit parameters.
    print_fit_parameters:bool
        Print all fit parameters

    Returns
    ----------
    (X,Y,Z) pcolormesh plot of Gaussian.

    '''

    xmin,xmax,dx=np.min(X),np.max(X),np.diff(X[0,:]).mean()
    ymin,ymax,dy=np.min(Y),np.max(Y),np.diff(Y[:,0]).mean()

    #PLOT
    fig,ax=plt.subplots(1,1,figsize=(9,6.5))
    S=ax.pcolormesh(X-dx/2,Y-dy/2,Z,cmap='viridis')

    ax.set_xlim(xmin,xmax-dx)
    ax.set_ylim(ymin,ymax-dy)

    cb=plt.colorbar(S,orientation='vertical')
    cb.set_label(label='Coincidences',size=16)
    cb.ax.tick_params(labelsize=16)

    if fit_plot:

        res=fit_gaussian((X,Y,Z))

        A,x0,y0,xsigma,ysigma,rho,k=res.x

        xt=(np.linspace(xmin,xmax-dx,150))
        yt=(np.linspace(ymin,ymax-dy,150))

        Xt,Yt=np.meshgrid(xt,yt)
        
        Zt=gauss2d([Xt,Yt],
                   A,x0,y0,xsigma,ysigma,rho,k)
        Ztheory=gauss2d([X,Y],
                   A,x0,y0,xsigma,ysigma,rho,k)

        V=A*np.array([np.exp(-2),np.exp(-0.5)])
        ax.contour(Xt,Yt,Zt,V,
                   colors='white',linestyles='solid',linewidths=0.4)
        if text_box: 
            ax.text(0.60,0.80,('Marginal fit\n'r'$\lambda_i$: %1.2f  $\sigma_i$ (rms): %1.3f'+'\n'
                +r'$\lambda_s$: %1.2f  $\sigma_s$(rms): %1.3f'+
                '\n'+r'$\rho$: %1.4f')
                 %(x0,xsigma,y0,ysigma,rho),bbox=dict(facecolor='white',alpha=0.5),transform=ax.transAxes)
    plt.show()

    if fit_plot:

        print('\nCentre: ',x0,y0,'\nSTD: ',xsigma,ysigma,'\nCorrelation: ',rho) 
        print('Chi2: ',2*res.cost)

    if print_fit_parameters:
        idler_gate_resolution=gate_resolution[0]
        signal_gate_resolution=gate_resolution[1]
        print_parameters(X,Y,Z,idler_gate_resolution,signal_gate_resolution)

def plot_tf_data(gs_data,
                 fit=False,text_box=True,
                 print_fit_parameters=False):
    '''
    Plot all time frequency data for GS algorithm. 

    Parameters
    ----------
    gs_data:tuple
        Wi: ndarray 
            2D meshgrid of Idler frequencies
        Ws: ndarray 
            2D array of Signal frequencies.     
        Ti: ndarray 
            2D array of Idler time     
        Ts: ndarray 
            2D array of Signal time     
        Iww: ndarray 
            2D array of Frequency-Frequency count data.     
        Itw: ndarray 
            2D array of Time-Frequency data.     
        Iwt: ndarray 
            2D array of Frequency-Time data.     
        Itt: ndarray 
            2D array of Time-Time data.     
    fit:bool
        Fit array to 2D Gaussian
    text_box:bool
        Show text box of fit parameters.
    print_fit_parameters:bool
        Print all fit parameters
    '''

    colormaps=['viridis','viridis','viridis','viridis']

    if len(gs_data)==8:
        Wi,Ws,Ti,Ts,Iww,Itw,Iwt,Itt=gs_data
        Ti1,Ts2=Ti,Ts
        Wi2,Ws1=Wi,Ws
    elif len(gs_data)==12:
        Wi,Ws,Iww,Ti1,Ws1,Itw,Wi2,Ts2,Iwt,Ti,Ts,Itt=gs_data

    fig,ax=plt.subplots(2,2,figsize=(12,8))
    ax=ax.reshape(-1)
    S0=ax[0].pcolormesh(Wi,Ws,Iww,cmap=colormaps[0])
    S1=ax[1].pcolormesh(Ti1,Ws1,Itw,cmap=colormaps[1])
    S2=ax[2].pcolormesh(Wi2,Ts2,Iwt,cmap=colormaps[2])
    S3=ax[3].pcolormesh(Ti,Ts,Itt,cmap=colormaps[3])
    [fig.colorbar([S0,S1,S2,S3][i],ax=ax[i]) for i in range(4)]  

    [ax[i].set_xlabel(r'$\omega_i$ (fs$^{-1}$)',fontsize=18) for i in [0,2]]
    [ax[i].set_ylabel(r'$\omega_s$ (fs$^{-1}$)',fontsize=18) for i in [0,1]]
    [ax[i].set_xlabel(r'$t_i$ (fs)',fontsize=18) for i in [1,3]]
    [ax[i].set_ylabel(r'$t_s$ (fs)',fontsize=18) for i in [2,3]]


    if fit:
        fit1=fit_gaussian((Wi,Ws,Iww))
        fit2=fit_gaussian((Ti1,Ws1,Itw))
        fit3=fit_gaussian((Wi2,Ts2,Iwt))
        fit4=fit_gaussian((Ti,Ts,Itt))

        for i in range(4):  
            A,x0,y0,xsigma,ysigma,rho,k=[fit1.x,fit2.x,fit3.x,fit4.x][i]
            X=[Wi,Ti1,Wi2,Ti][i]
            Y=[Ws,Ws1,Ts2,Ts][i]
            Z=[Iww,Itw,Iwt,Itt][i]

            xmin,xmax,dx=np.min(X),np.max(X),np.diff(X[0,:]).mean()
            ymin,ymax,dy=np.min(Y),np.max(Y),np.diff(Y[:,0]).mean()
            xt=(np.linspace(xmin,xmax-dx,150))
            yt=(np.linspace(ymin,ymax-dy,150))

            Xt,Yt=np.meshgrid(xt,yt)
            
            Zt=gauss2d([Xt,Yt],
                       A,x0,y0,xsigma,ysigma,rho,k)
           
            V=A*np.array([np.exp(-2),np.exp(-0.5)])
            ax[i].contour(Xt+dx/2,Yt+dy/2,Zt,V,
                       colors='white',linestyles='solid',linewidths=0.4)
            if text_box: 
                ax[i].text(0.5,0.70,('Marginal fit\n'+ r'$x_{0}$: %1.3f  $\sigma_x$ (rms): %1.4f'
                                      +'\n'+r'$y_{0}$: %1.3f  $\sigma_y$(rms): %1.4f'
                                      +'\n'+ r'$\rho$: %1.4f')%(x0,xsigma,y0,ysigma,rho),
                           bbox=dict(facecolor='white',alpha=0.5),transform=ax[i].transAxes)
    plt.tight_layout()
    plt.show()

    #print('Grid size: ', Ws.shape)
    print('dw:', np.diff(Wi)[0,0],np.diff(Ws.T)[0,0])
    print('dt:', np.diff(Ti)[0,0],np.diff(Ts.T)[0,0])
    print('Grid size:',Iww.shape,Itw.shape,Iwt.shape,Itt.shape)
    if fit:
        #DOF
        N1,N2,N3,N4=Iww.size-7,Itw.size-7,Iwt.size-7,Itt.size-7
        #FIT COST
        print('Fit cost:',2*fit1.cost/N1,2*fit2.cost/N2,2*fit3.cost/N3,2*fit4.cost/N4) 

def print_tf_fit_data(gs_data,
                      deconvolve=True,sample_size=100,
                      instrument_resolution=[[0.00018,0.00001],
                                                     [0.000335,0.00001],
                                                     [0.120,0.003],
                                                     [0.120,0.003]]):
    if len(gs_data)==8:
        Wi,Ws,Ti,Ts,Iww,Itw,Iwt,Itt=gs_data
        Ti1,Ts2=Ti,Ts
        Wi2,Ws1=Wi,Ws
    elif len(gs_data)==12:
        Wi,Ws,Iww,Ti1,Ws1,Itw,Wi2,Ts2,Iwt,Ti,Ts,Itt=gs_data

        idler_spectrometer_resolution=instrument_resolution[0]
        signal_spectrometer_resolution=instrument_resolution[1]

        idler_gate_resolution=instrument_resolution[2]
        signal_gate_resolution=instrument_resolution[3]

        print('\nPLOT 1') 
        print_parameters(Wi,Ws,Iww,idler_spectrometer_resolution,signal_spectrometer_resolution,
                        deconvolve=deconvolve,sample_size=sample_size)
        print('\nPLOT 2') 
        print_parameters(Ti1,Ws1,Itw,idler_gate_resolution,signal_spectrometer_resolution,
                        deconvolve=deconvolve,sample_size=sample_size)
        print('\nPLOT 3') 
        print_parameters(Wi2,Ts2,Iwt,idler_spectrometer_resolution,signal_gate_resolution,
                        deconvolve=deconvolve,sample_size=sample_size)
        print('\nPLOT 4') 
        print_parameters(Ti,Ts,Itt,idler_gate_resolution,signal_gate_resolution,
                        deconvolve=deconvolve,sample_size=sample_size)

def plot_gaussian(X,Y,Z,
                  fit=False,text_box=False): 
    '''
    PLOT 2D Gaussian. 

    Parameters
    ----------
    X: ndarray
        2D meshgrid for x-axis 
    Y: ndarray
        2D meshgrid for y-axis 
    Z: ndarray
        JSA of the source
    fit: bool,optional
        Option to fit to Gaussian function
    text_box: bool,optional
        Add text box on plot with fit parameters

    Returns
    ----------
    out: ndarray,optional
        Fit parameters of Gaussian fit 
    '''

    fig,ax=plt.subplots(1,1,figsize=(6,4))

    dx=np.diff(X[0,:])[0]
    dy=np.diff(Y[:,0])[0]

    C1=plt.pcolormesh(X-dx/2,Y-dy/2,Z)


    if fit:
        fitp=fit_gaussian((X,Y,Z))

        A,x0,y0,xsigma,ysigma,rho,k=fitp.x

        V=A*np.array([np.exp(-2),np.exp(-0.5)])
        plt.contour(X,Y,gauss2d([X,Y],*fitp.x),V,colors='white',linestyles='dashed',linewidths=0.4)

        if text_box: 
            ax.text(0.50,0.80,(r'$x_0$: %1.2f  $\sigma_x$ (rms): %1.4f'+'\n'
                +r'$y_0$: %1.2f  $\sigma_y$(rms): %1.4f'+
                '\n'+r'$\rho$: %1.4f')
                 %(x0,xsigma,y0,ysigma,rho),bbox=dict(facecolor='white',alpha=0.5),transform=ax.transAxes)
    ax.set_xlabel(r'$x$',fontsize=14)
    ax.set_ylabel(r'$y$',fontsize=14)
    fig.colorbar(C1,ax=ax)
    plt.show()

    if fit:
        return fitp