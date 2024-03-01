import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
from import_data import Data
from define_window import lmin, lmax, Lmin, Lmax, Window, short_window, large_window

#Selection of the trial________________________________________________________
n=0 #trial number, from 0 to 9, 10 corresponds to the average
U_raw=Window[n]
Nt, Nx, Ny = U_raw.shape

#temporal filtering____________________________________________________________
butt_filt = signal.butter(N=4, Wn=[5,20], btype='bandpass', fs=110, output='sos')
U_filt=signal.sosfiltfilt(butt_filt, U_raw, axis=0)

#Smoothing with a gaussian mask________________________________________________
U_smooth = np.zeros_like(U_raw)
for t in range(Nt):
    U_smooth[t]=gaussian_filter(U_filt[t], sigma=3, mode='constant')

pmax, pmin = np.max(U_smooth, axis=(0,1,2)), np.min(U_smooth, axis=(0,1,2))


#animation of the data_________________________________________________________
X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,8))
surf = [ax.plot_surface(X, Y, U_smooth[0] , cmap='seismic', vmin=pmin, vmax=pmax)]
ax.set_zlim(pmin, pmax)

def update(frame,surf):
    surf[0].remove()
    surf[0] = ax.plot_surface(X, Y, frame , cmap='seismic', vmin=pmin, vmax=pmax)
    return surf,

ani = animation.FuncAnimation(fig,update,fargs=(surf,),frames=U_smooth,interval=200,repeat_delay=2000)
plt.suptitle("Propagating wave")
plt.title("Average across trials" if n==10 else "trial n° "+str(n))
plt.show()


#saving the animation__________________________________________________________
# f = r"C:\Users\mmill\Desktop\surface.gif" 
# writergif = animation.PillowWriter(fps=10) 
#ani.save(f, writer=writergif)
