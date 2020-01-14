'''
###############################################################################
Ultrafast Quantum Optics Package
###############################################################################

Quantum Optics and Quantum Information Group
Written by 
> Jean-Philippe MacLean: jpmaclean@uwaterloo.ca
> Sacha Schwarz sacha.schwarz@uwaterloo.ca

'''

#Initialize modules 
from .analytics import * 
from .interface import * 
from .methods import * 
from .states import * 

#Define constants
c=0.299792458 #Speed of light in um/fs or mm/ps

import warnings
warnings.filterwarnings("ignore")

from timeit import default_timer as time