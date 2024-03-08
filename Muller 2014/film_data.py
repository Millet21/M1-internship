import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from scipy.ndimage import gaussian_filter
import scipy.signal as signal

from define_window import lmin, lmax, Lmin, Lmax, Window
#______________________________________________________________________________
Time=np.linspace(-100, 500, 66, endpoint=False)

trial_number = 0

U=Window[trial_number]

#Smoothing
if True:
    Smooth  = np.zeros_like(U)
    for i in range(len(U)):
        Smooth[i]=gaussian_filter(U[i], sigma=3)
    U=Smooth

u_max, u_min = np.max(U), np.min(U)

fig, ax = plt.subplots()

ax.set_title("trial n°"+str(trial_number))

u = ax.imshow(U[0], cmap='seismic', norm=plt.Normalize(vmin=u_min, vmax=u_max))
    
frame_text = fig.text(0.15, 0.75, '')#, ha='center')#, va='bottom')

def init():
    u.set_array(np.zeros((lmax-lmin, Lmax-Lmin)))
    return [u] 

def update(frame):
    u.set_array(U[frame])
    frame_text.set_text(str(round(Time[frame],1))+' ms')
    return [u, frame_text]

ani = anim.FuncAnimation(fig,func=update,frames=range(len(U)),init_func=init,interval=100, repeat_delay=1000, )

plt.show()


# f = r"C:\Users\mmill\Desktop\smoothed.gif" 
# writergif = anim.PillowWriter(fps=10) 
# ani.save(f, writer=writergif)
