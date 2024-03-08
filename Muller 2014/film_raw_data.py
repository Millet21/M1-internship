import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as colors
import numpy as np
import scipy.ndimage as img
import scipy.signal as signal

from import_data import Data
from define_window import lmin, lmax, Lmin, Lmax

#______________________________________________________________________________
Time=np.linspace(-100, 500, 66, endpoint=False)

trial_number = 0
U=np.copy(Data[trial_number])

U[U==0]=None #suppression of values out of camera range

masked = np.ma.masked_invalid(U)
u_max, u_min = np.max(masked), np.min(masked)

#______________________________________________________________________________
Filtering=False
if Filtering:
    butt_filt = signal.butter(N=8, Wn=[5,20], btype='bandstop', fs=110, output='sos')
    U=signal.sosfiltfilt(butt_filt, U, axis=0)
#______________________________________________________________________________
Smoothing = True
if Smoothing:
    Smooth  = np.zeros_like(U)
    for i in range(len(U)):
        Smooth[i]=img.gaussian_filter(U[i], sigma=2)
    U=Smooth
#______________________________________________________________________________
window_visual = True
if window_visual:
    edge_value=u_max  
    U[:,lmin:lmax+1, Lmin], U[:,lmin:lmax+1, Lmax],  U[:,lmin, Lmin:Lmax+1],  U[:,lmax, Lmin:Lmax+1] = edge_value, edge_value, edge_value, edge_value
#______________________________________________________________________________

fig, ax1 = plt.subplots()
ax1.set_title("trial n°"+str(trial_number))

u = ax1.imshow(U[0], cmap='jet', norm=colors.SymLogNorm(linthresh=.0005,vmin=u_min, vmax=u_max)) #best linthresh=.0002
frame_text = fig.text(0.15, 0.73, '')
fig.colorbar(u, ax=ax1)

def update(frame):
    u.set_array(U[frame])
    frame_text.set_text(str(round(Time[frame],1))+' ms')
    return [u, frame_text]

ani = anim.FuncAnimation(fig,func=update,frames=range(len(U)),interval=100,repeat_delay=1000)
plt.show()


