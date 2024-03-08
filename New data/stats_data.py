import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat

from import_single_data import Data, Label
#%%____________________________________________________________________________
N_trials, Nt, Nx, Ny = Data.shape

n_trial = 0

Sample = np.copy(Data[n_trial])
#%%____________________________________________________________________________

mean, median, std = np.nanmean(Sample), np.nanmedian(Sample), np.nanstd(Sample)
maxi, mini = np.nanmax(Sample), np.nanmin(Sample)

masked = np.ma.masked_invalid(Sample)

histo, edges = np.histogram(masked.compressed(), bins=200, density=True)
x=edges[:-1]
step=np.mean(np.diff(x))


def gauss_fit(t,N, moy, sig):
    return N/((2*np.pi)**.5*sig)*np.exp(-(t-moy)**2/(2*sig**2))

N=sum(histo*step)

Fit=gauss_fit(x, N, mean, std)


#%%
plt.title(Label[0]+' waves in '+Label[1]+" monkey, trial n°"+str(n_trial))
plt.plot(x,histo,label='data')
plt.plot(x,Fit,label='gaussian fit')
plt.hlines(0, x[0], x[-1], 'gray')
plt.vlines(0, 0, max(histo)*1.2, 'gray')
plt.legend()
plt.show()
#%%

unif_data = stat.norm.cdf(masked.compressed())#,loc=mean, scale=std)

unif_histo, edge = np.histogram(unif_data, bins=200, density=True)

plt.title(Label[0]+' waves in '+Label[1]+" monkey, trial n°"+str(n_trial))
plt.plot(x,unif_histo,label='uniformized data')
plt.hlines(0, x[0], x[-1], 'gray')
plt.vlines(0, 0, max(unif_histo)*1.2, 'gray')
plt.legend()
plt.show()








