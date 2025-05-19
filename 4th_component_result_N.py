import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap

# loading data for the phase diagramm
loaded_phasediagramm = np.load("4th_component_Phasediagram_N.npz")
N_particles = loaded_phasediagramm["N_particles_array"]
p_unitless = loaded_phasediagramm["p_array"]
Magnetisation_per_N = loaded_phasediagramm["Mag_per_N_array"]

print(p_unitless)
print(N_particles)
print(Magnetisation_per_N)

# special chemical potential
c_spec = 0.0005264653090365891

# calculate meshgrid
p_mesh, N_mesh = np.meshgrid(p_unitless, N_particles)

# Colormap 1
h = plt.pcolormesh(N_mesh, p_mesh, Magnetisation_per_N, cmap='viridis', shading='auto')
plt.scatter(N_mesh, p_mesh, label = "grid")
cbar = plt.colorbar(h, ticks=[0, 0.25, 0.5, 0.75, 1])
cbar.set_label("Magnetisation per N")
plt.title("Magnetisation per Particle (dimensionless)")
plt.ylabel('p')
plt.xlabel(r'N')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.show()