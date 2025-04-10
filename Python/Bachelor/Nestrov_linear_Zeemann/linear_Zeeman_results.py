import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap

# define interaction strenght
c_unitless = 0.0039354800621376855

# loading data for the phase diagramm
loaded_phasediagramm = np.load("Around_c_squared_N3.npz")
result = loaded_phasediagramm["array1"]
mu_unitless = loaded_phasediagramm["array2"]
p_unitless = loaded_phasediagramm["array3"]
Magnetisation = loaded_phasediagramm["array4"]
Magnetisation_per_M = loaded_phasediagramm["array5"]

# calculate meshgrid
p_mesh, mu_mesh = np.meshgrid(p_unitless, mu_unitless)

# fitting a line
def line(mu, a, b):
    return a * mu - b

#popt, pcov = curve_fit(line, mu_data, p_data)

mu_values = np.linspace(mu_mesh[0,0], mu_mesh[-1,-1], 1000)

# plotting the meshgrid with defining lines in black
h = plt.contourf(mu_mesh, p_mesh, result)
cbar = plt.colorbar(h, ticks=[0, 1, 2, 3, 4, 5])  # <-- Set your custom ticks here
cbar.ax.set_yticklabels(['Polar', 'Antiferro', 'Easy Axis', 'Easy Plane', 'Vacuum', 'Other'])  # Optional: custom labels
#plt.scatter(mu_mesh, p_mesh, label = "grid")
plt.title("Phasediagram dimensionless")
plt.ylabel('p')
plt.xlabel(r'$\mu$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.legend()
plt.show()

# plotting the meshgrid with defining lines in black
h = plt.contourf(mu_mesh, p_mesh, Magnetisation)
plt.colorbar()
#plt.scatter(mu_mesh, p_mesh, label = "grid")
plt.title("Magnetisation dimensionless")
plt.ylabel('p')
plt.xlabel(r'$\mu$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.legend()
plt.show()

# Define levels for more color bins (e.g., 20 levels from 0 to 1)
bounds = np.linspace(0, 1.025, 41)  # 20 color bins
cmap = get_cmap('viridis', len(bounds) - 1)
norm = BoundaryNorm(bounds, cmap.N)

# Filled contour plot with custom color resolution
h = plt.contourf(mu_mesh, p_mesh, Magnetisation_per_M, levels=bounds, cmap=cmap, norm=norm)
plt.contour(mu_mesh, p_mesh, Magnetisation_per_M, levels=bounds, colors='black', linewidths=0.1) # lines between the levels

# Colorbar with ticks and optional custom labels
cbar = plt.colorbar(h, ticks=[0, 0.25, 0.5, 0.75, 1])
cbar.set_label("Magnetisation per M")
plt.title("Magnetisation per Particle (dimensionless)")
plt.ylabel('p')
plt.xlabel(r'$\mu$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.show()