import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from import_data import Data
from define_window import lmin, lmax, Lmin, Lmax

trial_number = 0

U=Data[trial_number]

window_visual = True
#______________________________________________________________________________
if window_visual:
    edge_value=np.max(U)  
    U[:,lmin:lmax+1, Lmin], U[:,lmin:lmax+1, Lmax],  U[:,lmin, Lmin:Lmax+1],  U[:,lmax, Lmin:Lmax+1] = edge_value, edge_value, edge_value, edge_value
#______________________________________________________________________________

fig, ax = plt.subplots()

im0 = ax.imshow(U[0], cmap='coolwarm', norm='log')
cbar = fig.colorbar(im0)

ims = [ [ax.imshow(u, cmap='coolwarm', norm='log')] for u in U]

ani = animation.ArtistAnimation(fig, artists=ims,interval=50,repeat_delay=1000)

ax.set_title("raw data")

plt.show()
