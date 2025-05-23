import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# parameters
mu_crit = 1e-5

# creating the axis for the Phasediagram
p_values = np.linspace(-2, 2, 2000)*mu_crit
mu_values = np.linspace(0.5, 1.5, 2000)*mu_crit

# creating the mesh 
P, MU = np.meshgrid(p_values, mu_values)
P = P/mu_crit
MU = MU/mu_crit

# defining M
M_1 = P/MU

# making M between 1 and -1
M_new = np.where(M_1 > 1, 1, M_1)
M_new = np.where(M_new < -1, -1, M_new)

# the real M
M_real = np.copy(M_new)
mask = ((np.abs(M_real) != 1) & (MU < 1)) | \
       ((M_real == 1) & (MU + P < 2)) | \
       ((M_real == -1) & (MU - P < 2))
M_real[mask] = 0

# Plot only p/mu
plt.figure(figsize=(8, 6))
mesh = plt.pcolormesh(MU, P, M_1, shading='auto', cmap='viridis')
plt.xlabel('mu')
plt.ylabel('p')
plt.title('Magnetisation per particle')
plt.colorbar(mesh, label='M')
plt.show()

# Plot only p/mu
plt.figure(figsize=(8, 6))
mesh = plt.pcolormesh(MU, P, M_new, shading='auto', cmap='viridis')
plt.xlabel('mu')
plt.ylabel('p')
plt.title('Magnetisation per particle')
plt.colorbar(mesh, label='M')
plt.show()

# Plot only p/mu
plt.figure(figsize=(8, 6))
mesh = plt.pcolormesh(MU, P, M_real, shading='auto', cmap='viridis')
plt.xlabel('mu')
plt.ylabel('p')
plt.title('Magnetisation per particle')
plt.colorbar(mesh, label='M')
plt.show()