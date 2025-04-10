import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn 
import scipy.constants as const
from scipy.optimize import curve_fit

# loading data for the phase diagramm
loaded_phasediagramm = np.load("for_csquared.npz")
N_par = loaded_phasediagramm["array1"]
mu_values = loaded_phasediagramm["array2"]

# fiiting a line on the data
N_fit = np.linspace(0, 10, 1000)

def line(mu, a, b):
    return a * mu + b

popt, pcov = curve_fit(line, N_par, mu_values)

a = popt[0]
b = popt[1]

# showing the results
print("y-Achsenabschnitt = ", b)
print("mu_1N = ", mu_values[0])

# plotting the results
plt.plot(N_par, mu_values, color = "blue", linestyle = "None", marker = "+")
plt.plot(N_fit, line(N_fit, *popt), color = "blue")
plt.ylabel("N")
plt.xlabel(r"$\mu$")
plt.title(r"finding $c^2/4$")
plt.show()