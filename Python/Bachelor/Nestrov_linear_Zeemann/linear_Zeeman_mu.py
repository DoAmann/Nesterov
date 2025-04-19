import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn 
import scipy.constants as const
import time

start = time.time()
# properties of the grid
N_points = 2**11 # number of gridpoints

x = np.arange(- N_points // 2, N_points // 2, 1)
k_array = 2*np.pi*np.fft.fftfreq(N_points, 1)  # momentum grid

# define physical properties
N_par: int = 20 # number of particles
m: float = 1.44*10**(-25) # atom mass
l: float = 220e-6 # length of the system in meters
n = N_par / N_points # particle density
q = 0.00 # quadratic Zeeman shift units in Hz
p = 1 # linear Zeeman shift units in Hz
mu_zeeman = 1/1.155 * 1.530982e-33 # chemical potential
omega_parallel = 2* np.pi * 4 # Trapping frequency in longitudinal direction in Hz
omega_perp =  2 * np.pi * 400 #  Trapping frequency in transverse direction in Hz (isotropic in transverse direction)
a_B = const.physical_constants ["Bohr radius"] [0] # Bohr radius in meter
mu_Bohr = const.physical_constants["Bohr magneton"][0] # Bohr'sches Magneton
g_F = - 1/2 # Lande g-Faktor
a0 = 101.8*a_B #Scattering length 0 channel
a2 = 100.4*a_B #Scattering length 2 channel
dx = l  / N_points # distance of two points in meters 
sigma = N_points / (4 * np.sqrt(2)) # std of initial gaussian guess in unitless dimensions 

# calculating other parameters
Tscale = m * dx**2 / const.hbar
a_HO = np.sqrt(const.hbar/(m*omega_perp)) # harmonic oscillator length of transverse trap
c0 = 4*np.pi*const.hbar**2/(3*m)*(a0+2*a2)/(2*np.pi*a_HO**2) # density-density interactions

# transforming in unitless dimensions
c0 *= m *dx/ (const.hbar**2) # make unitless
c = c0  # already unitless, spin-spin interactions
c = 0
q *= 2 * np.pi * Tscale # making quadratic Zeeman shift dimensionless
p *= 2 * np.pi * Tscale # making linear Zeeman shift dimensionless
omega_parallel *= Tscale  #making frequency unitless
mu_zeeman *= Tscale / const.hbar # making chemical potential dimensionless 
c_squared_over_4 = 0.00022875236904247806 # chemical potential for 1 particle q=p=0
B_of_c_squared_over_4 = np.abs((c_squared_over_4 * const.hbar / Tscale) / (g_F * mu_Bohr))

# define potential
def V(x):
    return 1/2 * (omega_parallel * x) ** 2

Vx = V(x)
Lap = - k_array**2

# define initial wavefunction
def Psi_Initial_0(x):
    return np.exp(- x**2 / (2 * sigma**2))

def Psi_Initial_neg(x):
    return np.exp(- x**2 / (2 * sigma**2)) 

def Psi_Initial_pos(x):
    return np.exp(- x**2 / (2 * sigma**2)) 

# normalize initial wavefunctions
Psi_i_0 = Psi_Initial_0(x); Psi_i_neg = Psi_Initial_neg(x); Psi_i_pos = Psi_Initial_pos(x)

norm = np.sqrt( N_par / (np.sum(np.abs(Psi_i_0)**2 + np.abs(Psi_i_neg)**2 + np.abs(Psi_i_pos)**2)))
Psi_i_0 *= norm; Psi_i_neg *= norm; Psi_i_pos *= norm 

# initial iterate
Psi_0 = Psi_i_0; Psi_neg = Psi_i_neg; Psi_pos = Psi_i_pos
FT_Psi0_0 = fftn(Psi_0); FT_Psi0_neg = fftn(Psi_neg); FT_Psi0_pos = fftn(Psi_pos)
FT_Psi1_0 = FT_Psi0_0; FT_Psi1_neg = FT_Psi0_neg; FT_Psi1_pos = FT_Psi0_pos 

# calculating the absolute square
abs_Psi0_sq = np.abs(Psi_0)**2
abs_Psineg_sq = np.abs(Psi_neg)**2
abs_Psipos_sq = np.abs(Psi_pos)**2

# updating wavefunctions
rho = abs_Psi0_sq + abs_Psineg_sq + abs_Psipos_sq
    
#calculating interior of fouriertrafo
PsiX_0 = -Vx * Psi_0 - c * rho * Psi_0 + mu_zeeman * Psi_0 - c * (abs_Psipos_sq + abs_Psineg_sq) * Psi_0 - 2 * c * Psi_neg * np.conj(Psi_0) * Psi_pos
PsiX_neg = -Vx * Psi_neg - c * rho * Psi_neg + (- p - q + mu_zeeman) * Psi_neg - c * (abs_Psineg_sq + abs_Psi0_sq - abs_Psipos_sq) * Psi_neg - c * (Psi_0)**2 * np.conj(Psi_pos)
PsiX_pos = -Vx * Psi_pos - c * rho * Psi_pos + (p - q + mu_zeeman) * Psi_pos - c * (abs_Psipos_sq + abs_Psi0_sq - abs_Psineg_sq) * Psi_pos - c * (Psi_0)**2 * np.conj(Psi_neg)
    
FT_PsiX_0 = fftn(PsiX_0)
FT_PsiX_neg = fftn(PsiX_neg)
FT_PsiX_pos = fftn(PsiX_pos)

#iteration parameters
dt = 0.7; c_pre = 2;               #stepsize and parameter for preconditioner
Restart = 3000                     #for the condition of restarting
restarts = 0                    #for counting the number of restarts
ITER = 60000                     #number of maximal iterations
tol=10**(-11)                   #tolerance
jj = 0; ii = 0; i = 0 
e_0=1; e_neg=1; e_pos=1

# define lists to collect the errors
Error_0 = np.ones(ITER + 1); Error_neg = np.ones(ITER + 1); Error_pos = np.ones(ITER + 1)

P_inv =  (1/(c_pre - Lap))
while np.max([e_0, e_neg, e_pos])>tol and i < ITER:
    i += 1; ii += 1; jj += 1

    #iteration
    FT_Psi2_0 = (2 - 3/ii) * FT_Psi1_0 + dt**2 * P_inv * ((1/2) * Lap * FT_Psi1_0 + FT_PsiX_0) - (1 - 3/ii)*FT_Psi0_0
    Psi2_0 = ifftn(FT_Psi2_0)

    FT_Psi2_neg = (2 - 3/ii) * FT_Psi1_neg + dt**2 * P_inv * ((1/2) * Lap * FT_Psi1_neg + FT_PsiX_neg) - (1 - 3/ii)*FT_Psi0_neg
    Psi2_neg = ifftn(FT_Psi2_neg)

    FT_Psi2_pos = (2 - 3/ii) * FT_Psi1_pos + dt**2 * P_inv * ((1/2) * Lap * FT_Psi1_pos + FT_PsiX_pos) - (1 - 3/ii)*FT_Psi0_pos
    Psi2_pos = ifftn(FT_Psi2_pos)

    #gradient restart
    sum1 = np.sum((np.conj((1/2) * Lap * FT_Psi1_0 + FT_PsiX_0)) * (FT_Psi2_0 - FT_Psi1_0))
    sum2 = np.sum((np.conj((1/2) * Lap * FT_Psi1_neg + FT_PsiX_neg)) * (FT_Psi2_neg - FT_Psi1_neg))
    sum3 = np.sum((np.conj((1/2) * Lap * FT_Psi1_pos + FT_PsiX_pos)) * (FT_Psi2_pos - FT_Psi1_pos))

    cond1 = sum1 + sum2 + sum3
    if cond1 > 0 and ii > Restart:
        ii = 1
        restarts += 1

    # calculating the absolute square
    abs_Psi2_0_sq = np.abs(Psi2_0)**2
    abs_Psi2_neg_sq = np.abs(Psi2_neg)**2
    abs_Psi2_pos_sq = np.abs(Psi2_pos)**2
    
    # updating wavefunctions
    rho = abs_Psi2_0_sq + abs_Psi2_neg_sq + abs_Psi2_pos_sq
    
    #calculating interior of fouriertrafo
    PsiX_0 = -Vx * Psi2_0 - c * rho * Psi2_0 + mu_zeeman * Psi2_0 - c * (abs_Psi2_pos_sq + abs_Psi2_neg_sq) * Psi2_0 - 2 * c * Psi2_neg * np.conj(Psi2_0) * Psi2_pos
    PsiX_neg = -Vx * Psi2_neg - c * rho * Psi2_neg + (- p - q + mu_zeeman) * Psi2_neg - c * (abs_Psi2_neg_sq + abs_Psi2_0_sq - abs_Psi2_pos_sq) * Psi2_neg - c * (Psi2_0)**2 * np.conj(Psi2_pos)    
    PsiX_pos = -Vx * Psi2_pos - c * rho * Psi2_pos + (p - q + mu_zeeman) * Psi2_pos - c * (abs_Psi2_pos_sq + abs_Psi2_0_sq - abs_Psi2_neg_sq) * Psi2_pos - c * (Psi2_0)**2 * np.conj(Psi2_neg)
    
    FT_PsiX_0 = fftn(PsiX_0)
    FT_PsiX_neg = fftn(PsiX_neg)
    FT_PsiX_pos = fftn(PsiX_pos)

    # calculating residual error
    e_0 = np.sqrt( (1 / N_points) * np.sum(np.abs((1/2) * Lap * FT_Psi2_0 + FT_PsiX_0)**2))
    e_neg = np.sqrt( (1 / N_points) * np.sum(np.abs((1/2) * Lap * FT_Psi2_neg + FT_PsiX_neg)**2))
    e_pos = np.sqrt( (1 / N_points) * np.sum(np.abs((1/2) * Lap * FT_Psi2_pos + FT_PsiX_pos)**2))

    Error_0[i] = e_0       
    Error_neg[i] = e_neg
    Error_pos[i] = e_pos

    # updating wavefunctions
    FT_Psi0_0 = FT_Psi1_0; FT_Psi1_0 = FT_Psi2_0
    FT_Psi0_neg = FT_Psi1_neg; FT_Psi1_neg = FT_Psi2_neg
    FT_Psi0_pos = FT_Psi1_pos; FT_Psi1_pos = FT_Psi2_pos

Error_0 = Error_0[1:i + 1]
Error_neg = Error_neg[1:i + 1]
Error_pos = Error_pos[1:i + 1]

Iterations = i
x_Iter = np.arange(0, Iterations, 1)

# getting the physical wavefunction
Psi2_0_phys =  Psi2_0 / np.sqrt(dx)
Psi2_neg_phys = Psi2_neg / np.sqrt(dx)
Psi2_pos_phys = Psi2_pos / np.sqrt(dx)

#calculating the spin-components and magnetisation
F_perp = np.sqrt(2) * (np.conj(Psi2_0) * Psi2_pos + np.conj(Psi2_neg) * Psi2_0)
F_z = np.abs(Psi2_pos)**2 - np.abs(Psi2_neg)**2
Magnetisation = np.sum(F_z)
Magnetisation_per_N = Magnetisation / (np.sum(rho))

max_F_perp = np.max(np.abs(F_perp))
max_F_z = np.max(np.abs(F_z))

#calculating the number of particles
Nparticles = np.sum(rho)

#calculating the energy
if (c == 0):
    # calculating Laplace
    Lap_0 = ifftn( (1/2) * Lap * FT_Psi2_0)
    Lap_neg = ifftn( (1/2) * Lap * FT_Psi2_neg)
    Lap_pos = ifftn( (1/2) * Lap * FT_Psi2_pos)

    # calculating energy densities
    energy_0 = np.conj(Psi2_0) * Lap_0 + np.conj(Psi2_0) * (V(x) - mu_zeeman) * Psi2_0 # do I need the mu_zeeman?
    energy_neg = np.conj(Psi2_neg) * Lap_neg + np.conj(Psi2_neg) * (V(x) - mu_zeeman + p + q) * Psi2_neg
    energy_pos = np.conj(Psi2_pos) * Lap_pos + np.conj(Psi2_pos) * (V(x) - mu_zeeman - p + q) * Psi2_pos

    # calculating the energy
    energy_tot = np.real(energy_0 + energy_neg + energy_pos)
    Energy = np.sum(energy_tot)

# showing interesting results
print('Number of Iterations =', Iterations)
print('Number of Resets =', restarts)
print('Number of particles = ', Nparticles)
print("c dimensionless =", c)
print("Maximal density = ", np.max(rho))
print("Magnetsiation per particle = ", Magnetisation_per_N)
print("The Energy is =", Energy)
#print("Magnetic field of c^2/4 =", B_of_c_squared_over_4, "T")
#print(end_1 - start_1, end_2 - start_2)

# showing how long the code needed
end = time.time()
print("Time needed:", end - start)

if (Nparticles < 1):
    Psi2_0, Psi2_neg, Psi2_pos, F_perp, F_z, rho_max, Magnetisation, Magnetisation_per_N, Nparticles = (np.zeros(N_points),np.zeros(N_points),np.zeros(N_points),np.zeros(N_points),np.zeros(N_points),0,0,0,0)

# Plot of the results 
fig, ax = plt.subplots(2, 2, figsize=(14, 10)) 

# First Plot
ax[0,0].plot(x, np.abs(Psi2_0)**2, color='green', label= fr'$(\mu_z, p)=$ ({np.round(mu_zeeman, 8)}, {np.round(p, 8)})' + f"\n" + fr'$c =$ {np.round(c, 10)}' + f'\n' + f'q = {np.round(q, 8)}')
ax[0,0].plot(x, np.abs(Psi2_neg)**2, color='blue', label= r'$\Psi_{-1}$', linestyle = 'dashed')
ax[0,0].plot(x, np.abs(Psi2_pos)**2, color='red', label= r'$\Psi_{1}$', linestyle = 'dashdot')
ax[0,0].legend()
ax[0,0].set_title("Ground state") 
ax[0,0].set_xlabel("x")
ax[0,0].set_ylabel(rf"$|\Psi(x)|^2$")

# Second Plot
ax[0,1].plot(x_Iter, np.abs(Error_0), color='green', label=r"$\Psi_0$")
ax[0,1].plot(x_Iter, np.abs(Error_neg), color='blue', label=r"$\Psi_{-1}$", linestyle = 'dashed')
ax[0,1].plot(x_Iter, np.abs(Error_pos), color='red', label=r"$\Psi_{1}$", linestyle = 'dashdot')
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
ax[1,1].plot(x * dx, np.abs(Psi2_0_phys)**2, color='green', label= r'$\Psi_0$')
ax[1,1].plot(x * dx, np.abs(Psi2_neg_phys)**2, color='blue', label= r'$\Psi_{-1}$', linestyle = 'dashed')
ax[1,1].plot(x * dx, np.abs(Psi2_pos_phys)**2, color='red', label= r'$\Psi_{1}$', linestyle = 'dashdot')
ax[1,1].plot(x * dx, rho / dx, color = 'grey', label = r"$\rho$")
ax[1,1].legend()
ax[1,1].set_title("Real space groundstate") 
ax[1,1].set_xlabel("x")
ax[1,1].set_ylabel(rf"$|\Psi(x)|^2$")
plt.show()