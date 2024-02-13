import matplotlib.pyplot as plt
import numpy as np

from import_data import Data

#______________________________________________________________________________

lmin, lmax = 10, 60
Lmin, Lmax = 50, 175

Window={}
for k in range(10):
    Window['trial '+str(k+1)] = Data[k,:,lmin:lmax,Lmin:Lmax]
    

