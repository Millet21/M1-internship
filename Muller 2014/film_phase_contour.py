import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as colors
import scipy.ndimage as img
import scipy.signal as signal
import numpy as np
from import_data import Data
from define_window import Window, short_window
#______________________________________________________________________________
Time=np.linspace(-100, 500, 66, endpoint=False)
n_trial = 0

Sample = np.copy(Data[n_trial])

Nt, Nx, Ny = Sample.shape
#Removing the border___________________________________________________________
width_border = 5
kernel = np.zeros(shape=(2*width_border+1,2*width_border+1))
kernel[width_border,width_border]=1

Sample[Sample==0]=np.NAN
for k in range(Nt):
    Sample[k] = signal.convolve2d(Sample[k], kernel, mode='same')

nan_indices = np.isnan(Sample)

Sample[nan_indices]=0
#______________________________________________________________________________

z=signal.hilbert(Sample, axis=0)
Phi=np.angle(z)

#spatial smoothing
smoothing_coef = 5
Sample_smooth=np.zeros_like(Sample)
Phi_smooth=np.zeros_like(Phi)
for k in range(Nt):
    Sample_smooth[k]=img.gaussian_filter(Sample[k],sigma=smoothing_coef, mode='constant')#, cval=np.mean(Sample[k]))
    Phi_smooth[k]=img.gaussian_filter(Phi[k],sigma=smoothing_coef, mode='constant')#, cval=np.mean(Phi[k]))
   
#%%______________________________________________________________________________
U = Phi_smooth

U[nan_indices]=np.nan
u_min, u_max = np.nanmin(U), np.nanmax(U)

U_masked = np.ma.masked_invalid(U)

fig, ax1 = plt.subplots(figsize=(12,6))
ax1.set_title("trial n°" + str(n_trial))

u = ax1.imshow(U[0], cmap='hsv', norm=plt.Normalize(vmin=u_min, vmax=u_max))
phase_contour = ax1.contour(U_masked[0], levels=3, linewidths=.8)

frame_text = fig.text(0.15, 0.77, '')
cbar = fig.colorbar(u, ax=ax1, label='Phase (rad)')

def update(frame):
    ax1.clear()  
    u = ax1.imshow(U[frame], cmap='hsv', norm=plt.Normalize(vmin=u_min, vmax=u_max))
    phase_contour = ax1.contour(U_masked[frame], levels=3, linewidths=.8)  # Plot new contours
    frame_text.set_text(str(round(Time[frame], 1)) + ' ms')
    return [u,phase_contour, frame_text]

ani = anim.FuncAnimation(fig, func=update, frames=range(Nt), interval=200, repeat_delay=1000)
plt.show()


# f = r"C:\Users\mmill\Desktop\phase_contour.mp4" 
# writervideo = anim.FFMpegWriter(fps=60) 
# ani.save(f, writer=writervideo)



