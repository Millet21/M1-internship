import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as colors
import numpy as np
import scipy.ndimage as img
import scipy.signal as signal
import scipy.stats as stat

from import_single_data import Data, Label
#%%______________________________________________________________________________
N_trials, Nt, Nx, Ny = Data.shape

n_trial = 13

Sample = np.copy(Data[n_trial])
#______________________________________________________________________________
Filtered=True
if Filtered:
    butt_filt = signal.butter(N=4, Wn=[10], btype='lowpass', fs=110, output='sos')
    Sample=signal.sosfiltfilt(butt_filt, Sample, axis=0)
#______________________________________________________________________________
Smoothing = True
mean = np.nanmean(Sample)
mask = np.isnan(Sample)
Sample[mask]=0
if Smoothing:
    Smooth  = np.zeros_like(Sample)
    for i in range(Nt):
        Smooth[i]=img.gaussian_filter(Sample[i], sigma=3, mode='constant', cval=mean)
    Sample=Smooth
Sample[mask]=np.nan
#______________________________________________________________________________
Gauss_norm = True
if Gauss_norm:
    masked = np.ma.masked_invalid(Sample)
    unif_data = stat.norm.cdf(masked.compressed())
    Sample = np.reshape(unif_data, (Nt,Nx,Ny))
#______________________________________________________________________________

maxi, mini = np.nanmax(Sample), np.nanmin(Sample)

if not Gauss_norm:
    Norm = colors.SymLogNorm(linthresh=.1,vmin=mini, vmax=maxi)
else:
    Norm =colors.Normalize(vmin=mini, vmax=maxi)

condition = (' (Smoothed)' if (Smoothing and not Filtered) else '') + (' (Filtered)' if (Filtered and not Smoothing) else '') + (' (Smoothed and filtered)' if (Smoothing and Filtered) else '')

fig, ax1 = plt.subplots(figsize=(8,6))
fig.suptitle(Label[0]+' waves in '+Label[1]+" monkey, trial n°"+str(n_trial)+condition)

u = ax1.imshow(np.ones_like(Sample[0])*.5, cmap='seismic', norm=Norm)
fig.colorbar(u, ax=ax1, label='Standard Amplitude')

def update(frame):
    u.set_array(Sample[frame])
    ax1.set_title('frame n°'+str(frame))
    return [u]

ani = anim.FuncAnimation(fig,func=update,frames=range(Nt),interval=150,repeat_delay=1000)
plt.show()

# f = r"C:\Users\mmill\Desktop\unif2.gif" 
# writergif = anim.PillowWriter(fps=10) 
# ani.save(f, writer=writergif)

