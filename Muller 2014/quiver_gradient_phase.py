import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from scipy.ndimage import gaussian_filter
import scipy.signal as signal

from define_window import Window, short_window
#______________________________________________________________________________

Time=np.linspace(-100, 500, 66, endpoint=False)
dt=np.mean(np.diff(Time)) #ms

dx = 0.0566 #(mm/pixel)
dy=dx


n_trial = 0
Sample = np.copy(Window[n_trial])

Nt, Nx, Ny = Sample.shape

X, Y = np.meshgrid(np.arange(Ny), np.arange(Nx))

#Temporal filtering
if True:
    butt_filt = signal.butter(N=4, Wn=[5,20], btype='bandpass', fs=110, output='sos')
    Sample=signal.sosfiltfilt(butt_filt, Sample, axis=0)

#Analytic signal phase calculation
z=signal.hilbert(Sample, axis=0)
Phi=np.angle(z)

#Smoothing of the phase map
if True:
    Smooth  = np.zeros_like(Phi)
    for i in range(Nt):
        Smooth[i]=gaussian_filter(Phi[i], sigma=8,mode='constant',cval=np.mean(Phi[i]))
    Phi=Smooth


Wave_vector = -np.array(np.gradient(Phi, axis=(1,2)))

#Smoothing of the wave vector field
if False:
    ft  = np.fft.fft2(Wave_vector, axes=(2,3))
    ft_gauss = np.zeros_like(ft)
    for i in range(Nt):
        ft_gauss[0,i]=gaussian_filter(ft[0,i], sigma=10, mode='constant',cval=np.mean(ft[0,i]))
        ft_gauss[1,i]=gaussian_filter(ft[1,i], sigma=10, mode='constant',cval=np.mean(ft[1,i]))
    k_freq_smooth = np.fft.ifft2(ft_gauss, axes=(2,3))

#%% quiver plot with phase map_________________________________________________
U=Phi 
Q=Wave_vector

u_max, u_min = np.max(U), np.min(U)
n = 19

fig = plt.figure(figsize=(13,6), layout='constrained')

phase = plt.imshow(U[n], cmap='hsv')
cont = plt.contour(X,Y, U[n], levels=7, colors='black',linestyles='solid', linewidths=.7)
quiv = plt.quiver(X,Y, Q[1,n],Q[0,n], angles='xy')

cbar=fig.colorbar(phase, label='Phase (rad)')
cbar.add_lines(cont)
plt.show()


#%% Animation quiver + phase___________________________________________________
fig, (ax1) = plt.subplots(figsize=(10,6), layout='constrained')

fig.suptitle('Phase gradient animation')
ax1.set_title("trial n°"+str(n_trial))

quiver_anim = ax1.quiver(X,Y, Q[1,0], Q[0,0], angles='xy')
phase_anim = ax1.imshow(U[0], cmap='hsv', vmin=u_min, vmax=u_max)
contour_anim = plt.contour(X,Y, U[0], levels=7, colors='black',linestyles='solid', linewidths=.7)
frame_text = fig.text(0.5, 0.95, '')

cbar = fig.colorbar(phase_anim, ax=ax1)

def update(frame):
    ax1.clear()
    quiver_anim = ax1.quiver(X,Y, Q[1,frame], Q[0,frame], angles='xy')
    phase_anim = ax1.imshow(U[frame], cmap='hsv', vmin=u_min, vmax=u_max)
    contour_anim = plt.contour(X,Y, U[frame], levels=7, colors='black',linestyles='solid', linewidths=.7)
    frame_text.set_text(str(round(Time[frame],1))+' ms')
    cbar.add_lines(contour_anim)
    return [quiver_anim, phase_anim, contour_anim, frame_text]

ani = anim.FuncAnimation(fig,func=update,frames=range(Nt),interval=500, repeat_delay=1000)
 
plt.show()




