import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from scipy.ndimage import gaussian_filter
import scipy.signal as signal
from scipy.stats import linregress

from phase_latency import larg,Long,Latency

#%%
source = np.min(Latency)
l_s, L_s = np.where(Latency==source)

Dist = ((larg-larg[l_s,L_s])**2+(Long-Long[l_s,L_s])**2)**.5

list_dist = np.ravel(Dist)
list_latency = np.ravel(Latency)

slope, intercept, r, p, se  = linregress(list_dist, list_latency)

dmax = np.max(Dist)
ymax = slope*dmax+intercept 

plt.suptitle('Linear regression of the Phase latency with the distance')
plt.title('wave speed='+str(round(slope,2))+' m/s')
plt.plot(list_dist,list_latency , 'x')
plt.plot([0, dmax], [intercept, ymax], label='linear model, r²='+str(round(r,2)))
plt.xlabel('Distance from the source (mm)')
plt.ylabel('Latency (ms)')
plt.legend()
plt.show()