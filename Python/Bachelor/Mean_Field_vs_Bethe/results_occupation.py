import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Laden der gespeicherten Daten
data = np.load("Results_python/BEC_mu_Energy_data_new.npz")
mu_array = data["mu_vs_Npar_mu"]
N_par_array = data["N_par_mu"]
mu_array2 = data["mu2"]
p = data["p"]
Energy_per_N_values = data["Energy_per_N"]
result_array = data["result_array"]

# Alle eindeutigen p-Werte finden (nehmen wir aus dem mesh)
unique_p_values = np.unique(p)

# Dictionary, um Ergebnisse zu speichern
E_total_by_p = {}

for p_target in unique_p_values:
    # Filter rows mit gewünschtem p
    filtered = [row for row in result_array if row[1] == p_target]

    # Extrahiere E_per_N und N
    mu_values = np.array([row[0] for row in filtered])
    E_per_N = np.array([row[2] for row in filtered])
    N_array = np.array([row[3] for row in filtered])

    # calculate E_tot = E_per_N * N
    E_total = E_per_N * N_array

    # Speichern
    E_total_by_p[p_target] = (mu_values, N_array, E_total)

mu_vals1, N_vals1, E_vals1 = E_total_by_p[unique_p_values[0]]
mu_vals2, N_vals2, E_vals2 = E_total_by_p[unique_p_values[1]]
mu_vals3, N_vals3, E_vals3 = E_total_by_p[unique_p_values[2]]

# creating meshgrid
#mu_mesh, p_mesh = np.meshgrid(mu, p_array) 

# Erstellen des Plots
plt.figure(figsize=(8, 6))
plt.plot(mu_array, N_par_array, marker='o', linestyle= 'None', color='b', label = "calc")  
plt.xlabel(r'$\mu$')  # X-Achse: mu
plt.ylabel(r'$N_{\text{par}}$')  # Y-Achse: N_par
plt.title('Particle Number $N_{\text{par}}$ vs Chemical Potential $mu$')
plt.grid(True)
plt.legend()
plt.show()

# plotting E vs mu for different p
plt.figure(figsize=(8, 6))
plt.plot(N_vals1, E_vals1, marker='o', linestyle= '--', color='b', label = f"p = {unique_p_values[0]}") 
plt.plot(N_vals2, E_vals2, marker='o', linestyle= '--', color='red', label = f"p = {unique_p_values[1]}") 
plt.plot(N_vals3, E_vals3, marker='o', linestyle= '--', color='black', label = f"p = {unique_p_values[2]}")  
plt.xlabel('N')  # X-Achse: mu
plt.ylabel('E')  # Y-Achse: N_par
plt.title('Energy vs N')
plt.grid(True)
plt.legend()
plt.show()