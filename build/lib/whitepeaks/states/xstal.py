'''
###############################################################################
CRYSTAL MODULE
###############################################################################

This module contains the following classes:
> crystal()
  |-> BiBO()
  |-> SiO2()
  |-> BBO()
  |-> alphaBBO()

This module contains the following functions:
> GV(lambdas,fn_n,args)
> GVD(lambdas,fn_n,args)

'''

import numpy as np

from scipy.misc import derivative

c=0.299792458 #Speed of light in um/fs or mm/ps

class crystal:
    """Define different crystal parameters"""
    class BiBO:
        """Crystal axes for BiBO"""
        nx=[3.07403,0.03231,0.03163,-0.013376]
        ny=[3.16940,0.03717,0.03483,-0.01827]
        nz=[3.6545,0.05112,0.03713,-0.02261]
        def sellemeir(self,lambdas,args):
            A,B,C,D=args
            return (A+B/(lambdas**2-C)+D*lambdas**2)**0.5
        
        def n_e(self,lambdas,args):
            """Index of refraction for extraordinary axes"""
            theta,axes1,axes2=args
            theta*=np.pi/180
            
            #Index of refraction for the two axes
            n1=self.sellemeir(lambdas,axes1)
            n2=self.sellemeir(lambdas,axes2)
            
            return (np.cos(theta)**2/n1**2+np.sin(theta)**2/n2**2)**-0.5

    class SiO2:
        n=[0.6961663,0.0684043,0.4079426,0.1162414,0.8974794,9.896161]
        def sellemeir(self,lambdas,args):
            A,B,C,D,E,F=args
            return (1+A*lambdas**2/(lambdas**2-B**2)
                    +C*lambdas**2/(lambdas**2-D**2)
                    +E*lambdas**2/(lambdas**2-F**2))**0.5

    class BBO:
        no=[2.7359, 0.01878, 0.01822, 0.01354]
        ne=[2.3753, 0.01224, 0.01667, 0.01516]

        def sellemeir(self, lambdas,args):
            A,B,C,D=args
            return (A+B/(lambdas**2-C)-D*lambdas**2)**0.5

    class alphaBBO:
        no=[2.7471, 0.01878, 0.01822, 0.01354]
        ne=[2.37153, 0.01224, 0.01667, 0.01516]

        def sellemeir(self, lambdas,args):
            A,B,C,D=args
            return (A+B/(lambdas**2-C)-D*lambdas**2)**0.5

def GV(lambdas,fn_n,args):
    """
    Calculate the group velocity of light for a specified wavelength and
    crystal function. Specify the function for the index of refraction
    fn_n(lambda)as well as all other arguments the function requires.

    Parameters
    ----------
    lambdas: float 
       Wavelength in um.
    fn_n:function
       Function describing index of refraction as a function of
       wavelength,e.g., Sellemeir equations. 
    args:list
        Specifiy additional parameters required by fn_n

    Returns
    ----------
    out:float
        Group velocity at specified wavelength.
    """
    
    #index of refraction
    n=fn_n(lambdas,args)
    
    #first derivative evaluated at lambda
    dn=derivative(fn_n,lambdas,dx=1e-3,args=(args,),order=13)
    
    #return group velocity
    return (1.0/c*(n-lambdas*dn))**-1
    
def GVD(lambdas,fn_n,args):
    """
    Group Velocity Dispersion fs^2/um. Specify the function for the index
    of refraction fn_n(lambda)as well as all other arguments the function
    requires.

    Parameters
    ----------
    lambdas: float 
       Wavelength in um.
    fn_n:function
       Function describing index of refraction as a function of
       wavelength,e.g., Sellemeir equations. 
    args:list
        Specifiy additional parameters required by fn_n

    Returns
    ----------
    out:float
        Group velocity dispersion at specified wavelength.
    """
    
    #second derivative of index of refraction evaluated at lambda
    d2n=derivative(fn_n,lambdas,dx=1e-4,n=2,args=(args,),order=13)
    
    #return GVD
    return lambdas**3/(2*np.pi*c**2)*d2n
