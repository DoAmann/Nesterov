import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

#minimizing with discrete methode
a = 2e-2
b = 5e-3
s = 0.05
x0 = np.array([1,1])
y0 = x0
c = np.array([0,0])

#defenintion of the function to be minimized
def f(x1, x2, c):
    return a*(x1-c[0])**2 + b*(x2 - c[1])**2 

#definition of the function needed to minimize
def xk(y_k1, s):
    return y_k1 - s * np.array([2*a*(y_k1[0] - c[0]), 2*b*(y_k1[1] - c[1])])

def yk(xk, xk1, k):
    return xk + (k - 1)/(k+2) * (xk - xk1)

#defentition of the arrays
x_array = np.array([x0])
y_array = np.array([y0])

#beginning minimizing with while loop
k = 1
while (abs(f(x_array[-1,0], x_array[-1,1], c) - 0) > 1e-10 and k < 10000):
    x_k = xk(y_array[-1] , s)
    x_array = np.append(x_array, [x_k], axis = 0)

    y_k = yk(x_array[-1], x_array[-2], k)
    y_array = np.append(y_array, [y_k], axis = 0)
    
    k += 1
    k_max = k

#sorting and collecting x_1 and x_2 in seperate arrays
x1_coor_arr = []
x2_coor_arr = []
for vec in x_array:
    x1_coor_arr.append(vec[0])
    x2_coor_arr.append(vec[1])
x1_coord = np.array(x1_coor_arr)
x2_coord = np.array(x2_coor_arr)

#defining time parameter
t_param = np.arange(0, k_max, 1) * np.sqrt(s)

#calculating the error
f_err = np.array([f(x1_coord[coor], x2_coord[coor], c) - 0 for coor in np.arange(0, len(x1_coord))])

#plotting the coordinates
plt.plot(x1_coord, x2_coord)
plt.title('Trajectories')
plt.ylabel('x2')
plt.xlabel('x1')
plt.show()

#plotting the error
plt.plot(t_param, f_err)
plt.title('error')
plt.yscale('log')
plt.xlabel('t')
plt.ylabel('f - f*')
plt.xlim((0, 300))
plt.show()

#minimizing using the ODE
#definition of parameters
c = np.array([0,0]) #shifting the function f(x)
x0 = np.array([1,1])
v0 = np.array([0,0]) #initial conditions

#definition of the function to be minimized
def f(x1, x2, c):
    return a*(x1-c[0])**2 + b*(x2 - c[1])**2 

def partial_f(x1, x2, c):
    return 2*a*(x1 - c[0]) + 2*b*(x2 - c[1])

def partial_x1_f(x1, c):
    return 2*a*(x1 - c[0])

def partial_x2_f(x2, c):
    return 2*b*(x2 - c[1])

#definition of ODE first component
def dSdx_1(t1, S1):
    x1, v1 = S1
    return [v1, -3/t1*v1 - partial_x1_f(x1, c)]

x0_1 = x0[0]
v0_1 = v0[0]
S0_1 = (x0_1, v0_1)

t = np.linspace(1,300,1000)
sol1 = odeint(dSdx_1, y0 = S0_1 ,t=t, tfirst = True)

sol_x_1 = sol1.T[0]
sol_v_1 = sol1.T[1]

#definition of ODE first component
def dSdx_2(t2, S2):
    x2, v2 = S2
    return [v2, -(3/t2)*v2 - partial_x2_f(x2, c)]

x0_2 = x0[1]
v0_2 = v0[1]
S0_2 = (x0_2, v0_2)

sol2 = odeint(dSdx_2, y0 = S0_2 ,t=t, tfirst = True)

sol_x_2 = sol2.T[0]
sol_v_2 = sol2.T[1]

#calculating the error
f_err_ODE = np.array([f(sol_x_1[coor], sol_x_2[coor], c) - 0 for coor in np.arange(0, len(sol_x_2))])

#plotting the coordinates
plt.plot(sol_x_1, sol_x_2, color = 'blue', label = 'ODE')
plt.plot(x1_coord, x2_coord, color = 'red', label = 'scheme', marker = '+', linestyle = 'none', markersize = '0.8')
plt.title('Trajectories')
plt.ylabel('x2')
plt.xlabel('x1')
plt.legend()
plt.show()

#plotting the error
plt.plot(t, f_err_ODE, color = 'blue', label = 'ODE')
plt.plot(t_param, f_err, color = 'red', label = 'scheme', marker = '+', linestyle = 'none', markersize = '0.8')
plt.title('error')
plt.yscale('log')
plt.xlabel('t')
plt.ylabel('f - f*')
plt.xlim((0, 300))
plt.legend()
plt.show()


#Accelerated scheme
#defining the variables
a = 2e-2  #prefactor of x1
b = 5e-3  #prefactor of x2
s = 0.05  #stepwidth
x0 = np.array([1,1]) #initial guess
c = np.array([0,0])  #Minimum of the x^2 function

#defenintion of the function to be minimized
def f(x1, x2, c):
    return a*(x1-c[0])**2 + b*(x2 - c[1])**2 

#definition of the function needed to minimize
def xk(y_k1, s):
    return y_k1 - s * np.array([2*a*(y_k1[0] - c[0]), 2*b*(y_k1[1] - c[1])])

def yk(xk, xk1, k):
    return xk + (k - 1)/(k+2) * (xk - xk1)

#defentition of the arrays
y0 = np.copy(x0)
x_array = np.array([x0])
y_array = np.array([y0])

#defining velocity
def v(x_latest, x):
    return (x_latest - x)/s

#defining restarting condition
k_min = 10
def restart(x1, x2, v1, v2, k, k_min, c):
    grad_x1 = 2*a*(x1-c[0])
    grad_x2 = 2*b*(x2 - c[1])
    if (grad_x1*v1 + grad_x2*v2 > 0) and (k > k_min):
        return True
    else:
        return False

#beginning minimizing with while loop
k = 1
i = 0
while (abs(f(x_array[-1,0], x_array[-1,1], c) - 0) > 1e-10 and k < 10000):
    x_k = xk(y_array[-1] , s)
    x_array = np.append(x_array, [x_k], axis = 0)

    y_k = yk(x_array[-1], x_array[-2], k)
    y_array = np.append(y_array, [y_k], axis = 0)

    v1 = v(x_array[-2,0], x_array[-1, 0])
    v2 = v(x_array[-2,1], x_array[-1, 1])

    if restart(x_array[-1,0], x_array[-1,1], v1, v2, k, k_min, c):
        k = 1
        i += 1
        y_array[-1] = x_array[-1]  # new initial starting point
    else:    
        k += 1

#sorting and collecting x_1 and x_2 in seperate arrays
x1_coor_arr = []
x2_coor_arr = []
for vec in x_array:
    x1_coor_arr.append(vec[0])
    x2_coor_arr.append(vec[1])
x1_coord = np.array(x1_coor_arr)
x2_coord = np.array(x2_coor_arr)

#plotting the coordinates
plt.figure(figsize=(10, 5))
plt.plot(x1_coord, x2_coord, color = 'blue', label = 'accelerated')
plt.text(0.2, 0, f'Number of restarts = {i}')
plt.title('Trajectories')
plt.ylabel('x2')
plt.xlabel('x1')
plt.legend()
plt.show()