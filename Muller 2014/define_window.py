import matplotlib.pyplot as plt
import numpy as np

from import_data import Data

#______________________________________________________________________________

Dl=40
DL=110

x0, y0 = 8, 55

lmin, lmax = x0, x0+Dl #5,55 = quite good
Lmin, Lmax = y0, y0+DL #55, 165
 
Window = Data[:,:,lmin:lmax,Lmin:Lmax]

short_window = Data[:,:,5:55,60:185]
large_window = Data[:,:,0:85,15:240]

