import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from scipy.ndimage import gaussian_filter
import scipy.signal as signal

from define_window import Window, short_window
#______________________________________________________________________________
Time=np.linspace(-100, 500, 66, endpoint=False)
n_trial = 0
Sample = Window[n_trial]

Nt, Nx, Ny = Sample.shape

#temporal filtering
Filter=True
if Filter:
    butt_filt = signal.butter(N=4, Wn=[5,20], btype='bandpass', fs=110, output='sos')
    Sample=signal.sosfiltfilt(butt_filt, Sample, axis=0)

#analytic signal phase calculation
z=signal.hilbert(Sample, axis=0)
Env = np.absolute(z)
Phi=np.angle(z)

#spatial smoothing
Smooth=np.zeros_like(Sample)
Phi_smooth=np.zeros_like(Sample)
Env_smooth=np.zeros_like(Sample)
for k in range(len(Sample)):
    Smooth[k]=gaussian_filter(Sample[k],sigma=3, mode='constant', cval=np.mean(Sample[k]))
    Phi_smooth[k]=gaussian_filter(Phi[k],sigma=5, mode='constant', cval=np.mean(Phi[k]))
    Env_smooth[k]=gaussian_filter(Env[k],sigma=2, mode='constant', cval=np.mean(Env[k]))
    


#______________________________________________________________________________
Top = Phi
Bottom = Phi_smooth


top_max, top_min = np.max(Top), np.min(Top)
bot_max, bot_min = np.max(Bottom), np.min(Bottom)

fig, (ax1, ax2) = plt.subplots(2, 1)

top = ax1.imshow(Top[0], cmap='hsv', norm=plt.Normalize(vmin=top_min, vmax=top_max))
bot = ax2.imshow(Bottom[0], cmap='hsv', norm=plt.Normalize(vmin=top_min, vmax=top_max))
cont = ax2.contour(Bottom[0], levels=5, colors='black',linestyles='solid', linewidths=.7)

frame_text = fig.text(0.5, 0.95, '', ha='center', va='top')
fig.colorbar(top, ax=ax1)
cbar2 = fig.colorbar(bot, ax=ax2)


def update(frame, cbar2):
    top.set_array(Top[frame])
    ax2.clear()
    bot = ax2.imshow(Bottom[frame], cmap='hsv', norm=plt.Normalize(vmin=top_min, vmax=top_max))
    cont = ax2.contour(Bottom[frame], levels=5, colors='black',linestyles='solid', linewidths=.7)
    cbar2.add_lines(cont)
    frame_text.set_text(str(round(Time[frame],1))+' ms')
    return [top, bot, cont, frame_text]

ani = anim.FuncAnimation(fig,func=update,fargs=(cbar2,),frames=range(Nt),interval=600, repeat_delay=1000, )

plt.show()

