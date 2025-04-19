import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

geladene_daten = np.load("daten.npz")
result = geladene_daten["array1"]
q_unitless = geladene_daten["array2"]
c1_unitless = geladene_daten["array3"]

c1_mesh, q_mesh = np.meshgrid(c1_unitless, q_unitless)

q_line = np.linspace(0, 0.05, 100)

def line_theo(q):
    return - 1/(25 * 2) * q

def line_theo2(q):
    return - 1/(42.211867085 * 2) * q

# fitting line on phase transition parameters
def line_calc(q, m):
    return m * q

q_data = np.array([0.009343166303551607, 0.018751215706433445, 0.02809438200998505, 0.03750243141286688, 0.04691048081574869, 0.0563185302186305, 0.06572657962151235, 0.07513462902439416, 0.084542678427276]) # calculated q0 values for phase transition
c1_data = np.array([-0.0001119425440656373, -0.0002238850881312746, -0.00033582763219691187, -0.0004477701762625492, -0.0005597127203281865, -0.0006716552643938237, -0.0007835978084594611, -0.0008955403525250984, -0.0010074828965907358]) # corresponding values of c1

popt, pcov = curve_fit(line_calc, q_data, c1_data)
print("Steigung = ", popt[0])
print("Steigung_theo = ", -1 / (50))
print("Steigung_theo2 = ", -1 / (2 * 42.211867085))
print("Steigung/Steigung_theo =", -popt[0] * 50)


h = plt.contourf(q_mesh, c1_mesh, result)
plt.axvline(0, color = 'black')
plt.axhline(0, color = 'black')
#plt.plot(q_line, line_theo(q_line), color = "red")
plt.plot(q_line, line_theo2(q_line), color = "green")
plt.plot(q_line, line_calc(q_line, popt), color = "black")
plt.ylabel(r'$c_1$')
plt.xlabel('q')
plt.colorbar()
plt.show()