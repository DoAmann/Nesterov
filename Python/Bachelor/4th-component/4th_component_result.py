import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap

# loading data for the phase diagramm
loaded_phasediagramm = np.load("Results_python/4th_component_Phasediagram0.npz")
result = loaded_phasediagramm["array1"]
mu_unitless = loaded_phasediagramm["array2"]
p_unitless = loaded_phasediagramm["array3"]
Magnetisation_per_N = loaded_phasediagramm["array4"]
Magnetisation_per_N_0 = loaded_phasediagramm["array5"]
Magnetisation_per_N_1 = loaded_phasediagramm["array6"]

# special chemical potential
c_spec = 0.0005264653090365891

# calculate meshgrid
p_mesh, mu_mesh = np.meshgrid(p_unitless/c_spec, mu_unitless/c_spec)

# plotting the meshgrid with defining lines in black
h = plt.contourf(mu_mesh, p_mesh, result)
cbar = plt.colorbar(h, ticks=[0, 1, 2, 3, 4, 5])  # <-- Set your custom ticks here
cbar.ax.set_yticklabels(['Polar', 'Antiferro', 'Easy Axis', 'Easy Plane', 'Vacuum', 'Other'])  # Optional: custom labels
plt.scatter(mu_mesh, p_mesh, label = "grid")
plt.title("Phasediagram dimensionless")
plt.ylabel('p')
plt.xlabel(r'$\mu$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

# Define levels for more color bins (e.g., 20 levels from 0 to 1)
bounds = np.linspace(-0.0001, 1.025, 41)  # 20 color bins
cmap = plt.get_cmap('viridis', len(bounds) - 1,)
norm = BoundaryNorm(bounds, cmap.N)

# Colormap 1
h = plt.contourf(mu_mesh, p_mesh, Magnetisation_per_N, levels=bounds, cmap=cmap, norm=norm)
plt.contour(mu_mesh, p_mesh, Magnetisation_per_N, levels=bounds, colors='black', linewidths=0.1) # lines between the levels
plt.scatter(mu_mesh, p_mesh, label = "grid")
cbar = plt.colorbar(h, ticks=[0, 0.25, 0.5, 0.75, 1])
cbar.set_label("Magnetisation per N")
plt.title("Magnetisation per Particle (dimensionless)")
plt.ylabel('p')
plt.xlabel(r'$\mu$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.show()

# Colormap 2
bounds = np.linspace(-0.2, 0.2, 61)  # 20 color bins
cmap = plt.get_cmap('viridis', len(bounds) - 1,)
norm = BoundaryNorm(bounds, cmap.N)

h = plt.contourf(mu_mesh, p_mesh, Magnetisation_per_N_0, levels=bounds, cmap=cmap, norm=norm)
plt.contour(mu_mesh, p_mesh, Magnetisation_per_N_0, levels=bounds, colors='black', linewidths=0.1)
cbar = plt.colorbar(h, ticks=[-0.2, -0.1, 0, 0.1, 2])
cbar.set_label("Magnetisation per N")
plt.title("Magnetisation per Particle (dimensionless)")
plt.ylabel('p')
plt.xlabel(r'$\mu$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.show()

# Colormap 3
bounds = np.linspace(-0.1, 1.025, 41)  # 20 color bins
cmap = plt.get_cmap('viridis', len(bounds) - 1,)
norm = BoundaryNorm(bounds, cmap.N)

h = plt.contourf(mu_mesh, p_mesh, Magnetisation_per_N_1, levels=bounds, cmap=cmap, norm=norm)
plt.contour(mu_mesh, p_mesh, Magnetisation_per_N_1, levels=bounds, colors='black', linewidths=0.1)
cbar = plt.colorbar(h, ticks=[0, 0.25, 0.5, 0.75, 1])
cbar.set_label("Magnetisation per N")
plt.title("Magnetisation per Particle (dimensionless)")
plt.ylabel('p')
plt.xlabel(r'$\mu$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.show()