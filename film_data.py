import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.signal import convolve2d

from import_data import Data
from define_window import lmin, lmax, Lmin, Lmax, Window

#______________________________________________________________________________
n=2 #trial number
U_raw = Window['trial '+str(n)]

#Smoothing with a mask_________________________________________________________
a, b = 5,5 #mask dimensions
mask=np.ones((a,b))
mask_norm=mask/(a*b)

U_smooth = np.zeros_like(U_raw)
for t in range(len(U_raw)):
    U_smooth[t]=convolve2d(U_raw[t], mask_norm, mode='same')

pmax, pmin = np.max(U_smooth, axis=(0,1,2)), np.min(U_smooth, axis=(0,1,2))

#animation of the data_________________________________________________________
X, Y = np.meshgrid(np.arange(Lmax-Lmin), np.arange(lmax-lmin))
Zero = np.zeros((lmax-lmin,Lmax-Lmin))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6,6))
surf = [ax.plot_surface(X, Y, Zero , cmap='coolwarm', vmin=pmin, vmax=pmax)]
ax.set_zlim(pmin, pmax)

#text = ax.text(6, -1,1, '', ha='center')
#im, = [ax.imshow(np.zeros((lmax-lmin,Lmax-Lmin)), cmap='coolwarm', vmin=pmin, vmax=pmax)]

def update(frame,surf):
    surf[0].remove()
    surf[0] = ax.plot_surface(X, Y, frame , cmap='coolwarm', vmin=pmin, vmax=pmax)
    #text.set_text(f'Frame {i+1}')
    return surf,

ani = animation.FuncAnimation(fig, update,fargs=(surf,), frames=U_smooth,interval=50,repeat_delay=1000)

plt.suptitle("Propagating wave")
plt.title("trial n° "+str(n))
plt.show()
