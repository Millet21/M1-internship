import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

#Number of oscillators
N=6

# Kuramoto model parameters
omega = np.random.uniform(0,1,size=N)  # natural frequencies

K = 0.1  # coupling strength

# Initial phase values
theta_init = np.random.uniform(0, 2*np.pi, size=N)

# Function to compute the derivative of theta
def kuramoto(theta, t, omega, K):
    dtheta_dt = omega + K * np.sin(theta[:, np.newaxis] - theta[np.newaxis, :]).sum(axis=0)
    return dtheta_dt

# Time array
t = np.linspace(0, 100, 1000)

# Solve the ODE
theta_sol = odeint(kuramoto, theta_init, t, args=(omega, K))

#Kuramoto order index


# Plotting animation
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Kuramoto Model Simulation')

# Initialize points for oscillators
points, = ax.plot([], [],'o')

def update(frame):
    theta = theta_sol[frame]
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Update positions of the points
    points.set_data(x, y)
    
    return points, 

ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)

plt.show()
