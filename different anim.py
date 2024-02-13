import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
 
fig, axs = plt.subplots(2)
 
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
 
line1, = axs[0].plot(x, y1, color='blue')
line2, = axs[1].plot(x, y2, color='red')
 
def update(frame):
    line1.set_ydata(np.sin(x + frame / 100))
    line2.set_ydata(np.cos(x + frame / 100))
    return line1, line2
 
ani = FuncAnimation(fig, update, frames=range(100), blit=True)
plt.show()

