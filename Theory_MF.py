import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# parameters
mu_crit = 1e-5

# creating the axis for the Phasediagram
p_values = np.linspace(-2, 2, 1000)
mu_values = np.linspace(0.5, 1.5, 1000)

# creating the mesh 
P, MU = np.meshgrid(p_values, mu_values)
P = P
MU = MU
N_par = np.zeros_like(MU, dtype = float)

# defining M and N
M = P/MU

# making M between 1 and -1
M_new = np.where(M > 1, 1, M)
M_new = np.where(M_new < -1, -1, M_new)

# the real M
mask = ((np.abs(M_new) != 1) & (MU < 1)) | \
       ((M_new == 1) & (MU + P < 2)) | \
       ((M_new == -1) & (MU - P < 2))
M_new[mask] = 0

# particle number
mask_N = (np.abs(M_new) != 1)
mask_N1 = M_new == 1
mask_N2 = M_new == -1
N_par[mask_N] = MU[mask_N]
N_par[mask_N1] = (MU[mask_N1] + P[mask_N1])/2
N_par[mask_N2] = (MU[mask_N2] - P[mask_N2])/2
N_par[mask] = 0

# Theoretical borders
def line(mu, slope, b):
    return slope * mu + b

# theoretical parameters for the line
slope_upper, b_upper = -1, 2
slope_lower, b_lower = 1, -2

# theoretical mu axis
mu_theo = np.linspace(mu_values[0], 1, 10)

# Plot the Magnetisation
plt.figure(figsize=(8, 6))
mesh = plt.pcolormesh(MU, P, M_new, shading='auto', cmap='viridis')
plt.plot()
plt.plot(mu_theo, line(mu_theo, slope_upper, b_upper), label = r"$2\mu_c - \mu$", color = "black", linewidth = 0.5)
plt.plot(mu_theo, line(mu_theo, slope_lower, b_lower), label = r"$\mu - 2\mu_c$", color = "black", linewidth = 0.5)
plt.legend(loc = "upper right")
plt.xlabel('mu')
plt.ylabel('p')
plt.title('Magnetisation per particle')
plt.colorbar(mesh, label='M')
plt.show()

# Settings for plot the Number of particles
levels = np.linspace(0, np.max(N_par), 50)

# Plot the Number of particles
plt.figure(figsize=(8, 6))
mesh = plt.contourf(MU, P, N_par,levels, cmap='viridis')
cl = plt.contour(MU, P, N_par, levels=levels, colors='black', linewidths=0.5)
plt.plot()
plt.xlabel(r'$\mu$')
plt.ylabel('p')
plt.title('Number of particle')
plt.colorbar(mesh, label='N')
plt.show()
