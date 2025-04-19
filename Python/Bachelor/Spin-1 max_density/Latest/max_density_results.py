import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle

# loading data for the phase diagramm
loaded_phasediagramm = np.load("Phasendiagram_data.npz")
result = loaded_phasediagramm["array1"]
q_unitless = loaded_phasediagramm["array2"]
c1_unitless = loaded_phasediagramm["array3"]

# loading data for the phasetransition line
#with open("arrays.pkl", "rb") as f:
#    loaded_phasetransition_line = pickle.load(f) 

#q_data = loaded_phasetransition_line["q0_unitless_array"]
#c1_data = loaded_phasetransition_line["c1_unitless_array"]
#rho_max_unitless_array = loaded_phasetransition_line["rho_max_unitless_array"]

q_data = np.array([0.00841136, 0.0167752, 0.02523408, 0.03369296, 0.04215184, 0.05061072,
  0.05916464, 0.06762352, 0.0760824, 0.08454128])

c1_data = np.array([-0.00010075, -0.0002015,  -0.00030224, -0.00040299, -0.00050374, -0.00060449,
  -0.00070524, -0.00080599, -0.00090673, -0.00100748])

rho_max_unitless_array = q_data / (-2*np.array([0.98892615, 0.98613258, 0.98892615, 0.99032294, 0.99116101, 0.99171973,
  0.99371514, 0.99381491, 0.99389251, 0.99395459]) * c1_data )

# calculating the mean max density
rho_max_mean = np.mean(rho_max_unitless_array)

# create the meshgrid and q-axis for the plot
c1_mesh, q_mesh = np.meshgrid(c1_unitless, q_unitless)
q_line = np.linspace(0, 0.051, 100)

# define fitting line and theoretical line
def line_theo_max_density(q):
    return - 1 / (rho_max_mean * 2) * q

def line_numeric(q, m):
    return m * q

# fitting line on phase transition parameters
popt, pcov = curve_fit(line_numeric, q_data, c1_data)

# slope of the transition line numeric
slope_numeric = popt[0]

# showing some slopes
print("Steigung = ", slope_numeric)
print("Steigung mit homogener Dichte = ", -1 / (50))
print("Steigung_theo = ", - 1 / (2 * rho_max_mean))
print("Steigung/Steigung_theo =", -slope_numeric * (2 * rho_max_mean))

# plotting the meshgrid with defining lines in black
h = plt.contourf(q_mesh, c1_mesh, result)
plt.plot(q_line, line_numeric(q_line, slope_numeric), color = "red", label = "numeric")
plt.plot(q_line, line_theo_max_density(q_line), color = "black", linestyle = "dashed", label = "theo") # theoretical line for the phasetransition q = 2n_max|c_1|
plt.plot(q_data[:5], c1_data[:5], color = "red",linestyle = "None", marker = "+")
plt.title("Phasediagram dimensionless")
plt.axvline(0, color = 'black')
plt.axhline(0, color = 'black')
plt.text(-0.035, -5e-4, "easy-axis")
plt.text(-0.03, 5e-4, "AF")
plt.text(0.015, -5e-4, "easy-plane")
plt.text(0.02, 5e-4, "polar")
plt.ylabel(r'$c_1$')
plt.xlabel('q')
plt.colorbar()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(1)
plt.show()