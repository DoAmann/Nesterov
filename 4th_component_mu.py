import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn 
import scipy.constants as const

#properties of the grid
N_points = 2**11 #number of gridpoints
x = np.arange(- N_points // 2, N_points // 2, 1) # spatial grid
k_array = 2*np.pi*np.fft.fftfreq(N_points, 1)  # momentum grid
dk = np.abs(k_array[1] - k_array[0]) # spacing in momentum space

# define physical properties
N_par = 4
 # number of particles
m = 1.44*10**(-25) # atom mass
l = 220e-6 # length of the system in meters
n = N_par / N_points # particle density
omega_parallel = 2* np.pi * 4  # Trapping frequency in longitudinal direction in Hz
omega_perp =  2 * np.pi * 400 #  Trapping frequency in transverse direction in Hz (isotropic in transverse direction)
a_B = const.physical_constants ["Bohr radius"] [0] # Bohr radius in meter
a0 = 101.8*a_B #Scattering length 0 channel
a2 = 100.4*a_B #Scattering length 2 channel
dx = l  / (N_points) # distance of two points in meters 
sigma = N_points / (4 * np.sqrt(2)) # std of initial gaussian guess in unitless dimensions

# calculating other parameters
Tscale = m * dx**2 / const.hbar
a_HO = np.sqrt(const.hbar/(m*omega_perp)) # harmonic oscillator length of transverse trap
c = 4*np.pi*const.hbar**2/(3*m)*(a0+2*a2)/(2*np.pi*a_HO**2) # density-density interactions

# transforming in unitless dimensions
c *= m *dx/ (const.hbar**2) # make unitless
omega_parallel *= Tscale  #making frequency unitless

# defining dimensionless interaction strenght and linear zeeman shift and chemical potential
c = 0.017
p = 0.6*0.0005264653090365891
mu = 1.1 * 0.0005264653090365891

# define potential
def V(x):
    return (omega_parallel * x)**2

Vx = V(x) 
Lap = - k_array**2

# define lists to collect the errors
Error_11 = []; Error_12 = []; Error_21 = []; Error_22 = []

# define initial wavefunction
def Psi_Initial_11(x):
    return np.exp(- x**2 / (2 * sigma**2)) * (1.5 + 0.05 * np.random.random(N_points))

def Psi_Initial_12(x):
    return np.exp(- x**2 / (2 * sigma**2)) * (1 + 0.05 * np.random.random(N_points))

def Psi_Initial_21(x):
    return np.exp(- x**2 / (2 * sigma**2)) * (1 + 0.05 * np.random.random(N_points)) 

def Psi_Initial_22(x):
    return np.exp(- x**2 / (2 * sigma**2)) * (1 + 0.05 * np.random.random(N_points)) 

# normalize initial wavefunctions
Psi_i_11 = Psi_Initial_11(x); Psi_i_12 = Psi_Initial_12(x); Psi_i_21 = Psi_Initial_21(x); Psi_i_22 = Psi_Initial_22(x)

norm = np.sqrt( N_par / (np.sum(np.abs(Psi_i_11)**2 + np.abs(Psi_i_12)**2 + np.abs(Psi_i_21)**2 + np.abs(Psi_i_22)**2)))
Psi_i_11 *= norm; Psi_i_12 *= norm; Psi_i_21 *= norm; Psi_i_22 *= norm

# initial iterate
Psi_11 = Psi_i_11; Psi_12 = Psi_i_12; Psi_21 = Psi_i_21; Psi_22 = Psi_i_22
FT_Psi0_11 = fftn(Psi_11); FT_Psi0_12 = fftn(Psi_12); FT_Psi0_21 = fftn(Psi_21); FT_Psi0_22 = fftn(Psi_22)
FT_Psi1_11 = FT_Psi0_11; FT_Psi1_12 = FT_Psi0_12; FT_Psi1_21 = FT_Psi0_21; FT_Psi1_22 = FT_Psi0_22

abs_Psi_11_squared = np.abs(Psi_11)**2
abs_Psi_12_squared = np.abs(Psi_12)**2
abs_Psi_21_squared = np.abs(Psi_21)**2
abs_Psi_22_squared = np.abs(Psi_22)**2

rho = abs_Psi_11_squared + abs_Psi_12_squared + abs_Psi_21_squared + abs_Psi_22_squared

PsiX_11 = (mu + p - Vx) * Psi_11 - 2 * c * ((abs_Psi_11_squared + abs_Psi_21_squared + abs_Psi_12_squared) * Psi_11 + Psi_12 * np.conj(Psi_22) * Psi_21)
PsiX_12 = (mu - Vx) * Psi_12 - 2 * c * ((abs_Psi_11_squared + abs_Psi_22_squared + abs_Psi_12_squared) * Psi_12 + Psi_11 * np.conj(Psi_21) * Psi_22)
PsiX_21 = (mu - Vx) * Psi_21 - 2 * c * ((abs_Psi_11_squared + abs_Psi_21_squared + abs_Psi_22_squared) * Psi_21 + Psi_22 * np.conj(Psi_12) * Psi_11)
PsiX_22 = (mu - p - Vx) * Psi_22 - 2 * c * ((abs_Psi_21_squared + abs_Psi_12_squared + abs_Psi_22_squared) * Psi_22 + Psi_21 * np.conj(Psi_11) * Psi_12)

FT_PsiX_11 = fftn(PsiX_11); FT_PsiX_12 = fftn(PsiX_12); FT_PsiX_21 = fftn(PsiX_21); FT_PsiX_22 = fftn(PsiX_22) 

#iteration parameters
dt = 0.55; c_pre = 4;               #stepsize and parameter for preconditioner
Restart = 2000                     #for the condition of restarting
restarts = 0                    #for counting the number of restarts
ITER = 90000                     #number of maximal iterations
tol=10**(-12)                   #tolerance
tol_mu = 1e-4                   # tolerance for computing mu
abs_max_sq_11 = 100; abs_max_sq_12 = 100; abs_max_sq_21 = 100; abs_max_sq_22 = 100
jj = 0; ii = 0; i = 0 
e_11=1; e_12=1; e_21=1; e_22=1

P_inv =  (1/(c_pre - Lap))

# beginning the while loop 
while np.max([e_11, e_12, e_21, e_22])>tol and i < ITER:
    i += 1; ii += 1; jj += 1

    #iteration
    FT_Psi2_11 = (2 - 3/ii) * FT_Psi1_11 + dt**2 * P_inv * (Lap * FT_Psi1_11 + FT_PsiX_11) - (1 - 3/ii)*FT_Psi0_11
    Psi2_11 = ifftn(FT_Psi2_11)

    FT_Psi2_12 = (2 - 3/ii) * FT_Psi1_12 + dt**2 * P_inv * (Lap * FT_Psi1_12 + FT_PsiX_12) - (1 - 3/ii)*FT_Psi0_12
    Psi2_12 = ifftn(FT_Psi2_12)

    FT_Psi2_21 = (2 - 3/ii) * FT_Psi1_21 + dt**2 * P_inv * (Lap * FT_Psi1_21 + FT_PsiX_21) - (1 - 3/ii)*FT_Psi0_21
    Psi2_21 = ifftn(FT_Psi2_21)

    FT_Psi2_22 = (2 - 3/ii) * FT_Psi1_22 + dt**2 * P_inv * (Lap * FT_Psi1_22 + FT_PsiX_22) - (1 - 3/ii)*FT_Psi0_22
    Psi2_22 = ifftn(FT_Psi2_22)

    # calculating the new squares
    abs_Psi2_11_squared = np.abs(Psi2_11)**2
    abs_Psi2_12_squared = np.abs(Psi2_12)**2
    abs_Psi2_21_squared = np.abs(Psi2_21)**2
    abs_Psi2_22_squared = np.abs(Psi2_22)**2

    #gradient restart
    sum1 = np.sum((np.conj(Lap * FT_Psi1_11 + FT_PsiX_11)) * (FT_Psi2_11 - FT_Psi1_11))
    sum2 = np.sum((np.conj(Lap * FT_Psi1_12 + FT_PsiX_12)) * (FT_Psi2_12 - FT_Psi1_12))
    sum3 = np.sum((np.conj(Lap * FT_Psi1_21 + FT_PsiX_21)) * (FT_Psi2_21 - FT_Psi1_21))
    sum4 = np.sum((np.conj(Lap * FT_Psi1_22 + FT_PsiX_22)) * (FT_Psi2_22 - FT_Psi1_22))

    cond1 = sum1 + sum2 + sum3 + sum4
    if cond1 > 0 and ii > Restart:
        ii = 1
        restarts += 1

    rho = abs_Psi2_11_squared + abs_Psi2_12_squared + abs_Psi2_21_squared + abs_Psi2_22_squared

    # Updating the PsiX terms
    PsiX_11 = (mu + p - Vx) * Psi2_11 - 2 * c * ((abs_Psi2_11_squared + abs_Psi2_21_squared + abs_Psi2_12_squared) * Psi2_11 + Psi2_12 * np.conj(Psi2_22) * Psi2_21)
    PsiX_12 = (mu - Vx) * Psi2_12 - 2 * c * ((abs_Psi2_11_squared + abs_Psi2_22_squared + abs_Psi2_12_squared) * Psi2_12 + Psi2_11 * np.conj(Psi2_21) * Psi2_22)
    PsiX_21 = (mu - Vx) * Psi2_21 - 2 * c * ((abs_Psi2_11_squared + abs_Psi2_21_squared + abs_Psi2_22_squared) * Psi2_21 + Psi2_22 * np.conj(Psi2_12) * Psi2_11)
    PsiX_22 = (mu - p - Vx) * Psi2_22 - 2 * c * ((abs_Psi2_21_squared + abs_Psi2_12_squared + abs_Psi2_22_squared) * Psi2_22 + Psi2_21 * np.conj(Psi2_11) * Psi2_12)

    FT_PsiX_11 = fftn(PsiX_11); FT_PsiX_12 = fftn(PsiX_12); FT_PsiX_21 = fftn(PsiX_21); FT_PsiX_22 = fftn(PsiX_22)

    # calculating the error
    e_11 = np.sqrt((1/N_points) * np.sum(np.abs(Lap * FT_Psi2_11 + FT_PsiX_11)**2))
    e_12 = np.sqrt((1/N_points) * np.sum(np.abs(Lap * FT_Psi2_12 + FT_PsiX_12)**2))
    e_21 = np.sqrt((1/N_points) * np.sum(np.abs(Lap * FT_Psi2_21 + FT_PsiX_21)**2))
    e_22 = np.sqrt((1/N_points) * np.sum(np.abs(Lap * FT_Psi2_22 + FT_PsiX_22)**2))

    Error_11.append(e_11)
    Error_12.append(e_12)
    Error_21.append(e_21)
    Error_22.append(e_22)

    # updating wavefunctions
    FT_Psi0_11 = FT_Psi1_11; FT_Psi1_11 = FT_Psi2_11
    FT_Psi0_12 = FT_Psi1_12; FT_Psi1_12 = FT_Psi2_12
    FT_Psi0_21 = FT_Psi1_21; FT_Psi1_21 = FT_Psi2_21
    FT_Psi0_22 = FT_Psi1_22; FT_Psi1_22 = FT_Psi2_22

Iterations = i
x_Iter = np.arange(0, Iterations, 1)

# calculating number of particles
Nparticles = np.sum(rho)

# defining the physical components
Psi_pos = Psi2_11; Psi_neg = Psi2_22; Psi_0 = (1 / np.sqrt(2)) * (Psi2_12 + Psi2_21); eta_0 = (1 / np.sqrt(2)) * (Psi2_12 - Psi2_21) 

#calculating spin components like on the poster
F_x = (1 / np.sqrt(2)) * (np.conj(Psi_pos) * (Psi_0 + eta_0) + np.conj(Psi_0) * (Psi_pos + Psi_neg) + np.conj(Psi_neg) * (Psi_0 + eta_0) + np.conj(eta_0) * (Psi_neg + Psi_neg))
F_y = (1j / np.sqrt(2)) *(np.conj(Psi_pos) * (Psi_0 + eta_0) + np.conj(Psi_0) * (- Psi_pos + Psi_neg) + np.conj(Psi_neg) * (- Psi_0 + eta_0) - np.conj(eta_0) * (Psi_neg + Psi_neg))
F_z = np.abs(Psi_pos)**2 - np.conj(Psi_0)*eta_0 - np.conj(eta_0) * Psi_0 - np.abs(Psi_neg)**2
Magnetisation_per_N_0 = np.sum(F_z) / Nparticles # Magnetisation from the F_z like in the poster
Magnetisation_per_N_1 = np.sum(np.abs(Psi_pos)**2 - np.abs(Psi_neg)**2) / Nparticles # Magnetisation like 3 component system

F_perp = np.real(F_x + 1j * F_y)
F_z = np.real(F_z)

# calculating spin components with pauli matrices
F_x = np.conj(Psi2_11) * Psi2_21 + np.conj(Psi2_21) * Psi2_11 + np.conj(Psi2_12) * Psi2_22 + np.conj(Psi2_22) * Psi2_12
F_y = 1j * (np.conj(Psi2_21) * Psi2_11 - np.conj(Psi2_11) * Psi2_21 - np.conj(Psi2_12) * Psi2_22 + np.conj(Psi2_22) * Psi2_12)
F_z = np.conj(Psi2_11) * Psi2_11 + np.conj(Psi2_12) * Psi2_12 - np.conj(Psi2_21) * Psi2_21 - np.conj(Psi2_22) * Psi2_22
Magnetisation_per_N = np.sum(F_z) / Nparticles # Magnetisation with pauli matrices

F_perp = np.real(F_x + 1j * F_y)
F_z = np.real(F_z)

# showing interesting results
print('Number of Iterations =', Iterations)
print('Number of Resets =', restarts)
print('Number of particles = ', np.sum(rho))
print("c unitless =", c)
print("Maximal density = ", np.max(rho))
print("chemical potential mu = ", mu)

# Plot of the results 
fig, ax = plt.subplots(2, 2, figsize=(15, 10)) 

# First Plot
ax[0,0].plot(x, rho, color = 'grey', label = r"$\rho$")
ax[0,0].plot(x, np.abs(Psi2_11)**2, color='green', label= f'$(\\mu, p)=$ ({np.round(mu, 8)}, {np.round(p,8)}) \n' + fr'$c =$ {np.round(c, 10)}')
ax[0,0].plot(x, np.abs(Psi2_12)**2, color='blue', label= r'$\Psi_{12}$', linestyle = 'dashed')
ax[0,0].plot(x, np.abs(Psi2_21)**2, color='purple', label= r'$\Psi_{21}$', linestyle = 'dashdot')
ax[0,0].plot(x, np.abs(Psi2_22)**2, color='red', label= r'$\Psi_{22}$', linestyle = '--')
ax[0,0].legend()
ax[0,0].set_title("Ground state") 
ax[0,0].set_xlabel("x")
ax[0,0].set_ylabel(rf"$|\Psi(x)|^2$")

# Second Plot
ax[0,1].plot(x_Iter, np.abs(Error_11), color='green', label=r"$\Psi_{11}$")
ax[0,1].plot(x_Iter, np.abs(Error_12), color='blue', label=r"$\Psi_{12}$", linestyle = 'dashed')
ax[0,1].plot(x_Iter, np.abs(Error_21), color='purple', label=r"$\Psi_{21}$", linestyle = 'dashdot')
ax[0,1].plot(x_Iter, np.abs(Error_22), color='red', label=r"$\Psi_{22}$", linestyle = '--')
ax[0,1].legend()
ax[0,1].set_yscale('log')
ax[0,1].set_title("Residual error") 
ax[0,1].set_xlabel("Iterations")
ax[0,1].set_ylabel("Error")

# Third plot
ax[1,0].plot(x, np.abs(F_perp), color='black', label=r"$|F_{\perp}|$")
ax[1,0].plot(x, np.abs(F_z), color='green', label=r"$|F_{z}|$", linestyle = 'dashed')
ax[1,0].legend()
ax[1,0].set_title("Spin") 
ax[1,0].set_ylabel(r"$F_\nu$")
ax[1,0].set_xlabel("x")

# Fourth plot
ax[1,1].plot(x, rho, color = 'grey', label = r"$\rho$")
ax[1,1].plot(x, np.abs(Psi_pos)**2, color='green', label= r'$\Psi_{1}$')
ax[1,1].plot(x, np.abs(Psi_0)**2, color='blue', label= r'$\Psi_{0}$', linestyle = 'dashed')
ax[1,1].plot(x, np.abs(Psi_neg)**2, color='red', label= r'$\Psi_{-1}$', linestyle = 'dashdot')
ax[1,1].plot(x, np.abs(eta_0)**2, color='purple', label= r'$\eta_{0}$', linestyle = 'dotted')
ax[1,1].legend()
ax[1,1].set_title(r"Groundstate in $(\Psi_1, \Psi_0, \Psi_{-1}, \eta_0)$") 
ax[1,1].set_xlabel("x")
ax[1,1].set_ylabel(rf"$|\Psi(x)|^2$")

plt.show()