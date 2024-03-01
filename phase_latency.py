import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from scipy.ndimage import gaussian_filter
import scipy.signal as signal

from define_window import Window, short_window
#______________________________________________________________________________

Time=np.linspace(-100, 500, 66, endpoint=False) #stim from 0 to 50ms
dt=np.mean(np.diff(Time)) #ms

dx = 0.0566 #(mm/pixel)
dy=dx

n_trial = 0
Sample = Window[n_trial]

Nt, Nx, Ny = Sample.shape
larg, Long = np.meshgrid( np.linspace(0,round(Ny*dy,1),Ny), np.linspace(0,round(Nx*dx,1),Nx) )

#Temporal filtering
butt_filt = signal.butter(N=4, Wn=[5,20], btype='bandpass', fs=110, output='sos')
Filtered=signal.sosfiltfilt(butt_filt, Sample, axis=0)

#Analytic signal phase calculation
z_raw=signal.hilbert(Sample, axis=0)
z_filt = signal.hilbert(Filtered, axis=0)


########## Choose the signal (raw or temporaly filtered)
show_z=z_filt 
########## Choose the time point from which the latency is computed
N0 = 19
##########
T0=Time[N0]
show_Phi=np.angle(show_z)
Phase = show_Phi[N0]

def consec(L):
    for k in range(len(L-1)):
        if -np.pi/2<L[k]<0 and np.pi/2>L[k+1]>0:
            return np.array([k, k+1])
#%% Calculation of the Phase Latency Map

Latency = np.zeros_like(Phase)
for i in range(Nx):
    for j in range(Ny):
        N1, N2 = consec(show_Phi[N0:, i, j])+N0
        omega = np.angle(show_z[N2,i,j]*np.conj(show_z[N1,i,j]))/dt
        tau = Time[N1]- show_Phi[N1, i, j]/omega
        latency = tau-T0
        Latency[i, j]=latency

Latency=gaussian_filter(Latency, sigma=4, mode='constant', cval=np.mean(Latency))

#%% Visualisation
if False:
    
    fig, ax1 = plt.subplots(figsize=(12,6), layout='constrained')
    
    plt.suptitle("Phase latency map, "+("Average across trials" if n_trial==10 else "trial n° "+str(n_trial)))
    ax1.set_title('+'+str(round(Time[N0],1))+' ms')
    
    phase_lat_map = ax1.pcolormesh(larg, Long, np.flipud(Latency), cmap='jet')
    phase_contour = ax1.contour(larg, Long, np.flipud(Latency), levels=10, colors='black', linewidths=.8)
    
    ax1.set_aspect(aspect='equal')
    
    ax1.set_ylabel('largeur (mm)')
    ax1.set_xlabel('Longueur (mm)')
    
    cbar = fig.colorbar(phase_lat_map, ax=ax1, label='Latency (ms)')
    cbar.add_lines(phase_contour)
    
    plt.show()
    
