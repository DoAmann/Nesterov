import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn 
import scipy.constants as const

# settings
trap = True

#properties of the grid
N_points = 2**11 #number of gridpoints
x = np.arange(- N_points // 2, N_points // 2, 1) # spatial grid
k_array = 2*np.pi*np.fft.fftfreq(N_points, 1)  # momentum grid
dk = np.abs(k_array[1] - k_array[0]) # spacing in momentum space

# define physical properties
N_par = 800
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

# defining dimensionless interaction strenght and linear zeeman shift
c = 0.017
c_spec = 0.0005264653090365891
p = 0.001

# define potential
if trap:
    def V(x):
        return (omega_parallel * x)**2
else:
    def V(x):
        return 0*x

Vx = V(x) 
Lap = - k_array**2

# define lists to collect the errors
Error_11 = []; Error_12 = []; Error_21 = []; Error_22 = []


if (all(Vx) == 0):

    # define initial wavefunction
    def Psi_Initial_pos(x):
        return np.sqrt(N_par / (4 * N_points)) * np.ones(N_points) * (1.0 + 0.1 * np.random.random(N_points))

    def Psi_Initial_0(x):
        return np.sqrt(N_par / (4 * N_points)) * np.ones(N_points) * (1.0 + 0.1 * np.random.random(N_points))

    def Psi_Initial_neg(x):
        return np.sqrt(N_par / (4 * N_points)) * np.ones(N_points) * (1.0 + 0.1 * np.random.random(N_points))

    def eta_Initial_0(x):
        return np.sqrt(N_par / (4 * N_points)) * np.ones(N_points) * (1.0 + 0.1 * np.random.random(N_points))
    
else:
    # define initial wavefunction
    def Psi_Initial_pos(x):
        return np.exp(- x**2 / (2 * sigma**2)) * (1 + 0.1 * np.random.random(N_points))

    def Psi_Initial_0(x):
        return np.exp(- x**2 / (2 * sigma**2)) * (1.1 + 0.1 * np.random.random(N_points))

    def Psi_Initial_neg(x):
        return np.exp(- x**2 / (2 * sigma**2)) * (1.2 + 0.1 * np.random.random(N_points)) 

    def eta_Initial_0(x):
        return np.exp(- x**2 / (2 * sigma**2)) * (1.1 + 0.1 * np.random.random(N_points)) 

# normalize initial wavefunctions
Psi_i_pos = Psi_Initial_pos(x); Psi_i_0 = Psi_Initial_0(x); Psi_i_neg = Psi_Initial_neg(x); eta_i_0 = eta_Initial_0(x)

norm = np.sqrt( N_par / (np.sum(np.abs(Psi_i_pos)**2 + np.abs(Psi_i_0)**2 + np.abs(Psi_i_neg)**2 + np.abs(eta_i_0)**2)))
Psi_i_pos *= norm; Psi_i_0 *= norm; Psi_i_neg *= norm; eta_i_0 *= norm

# initial iterate
Psi_pos = Psi_i_pos; Psi_0 = Psi_i_0; Psi_neg = Psi_i_neg; eta_0 = eta_i_0
FT_Psi0_pos = fftn(Psi_pos); FT_Psi0_0 = fftn(Psi_0); FT_Psi0_neg = fftn(Psi_neg); FT_eta0_0 = fftn(eta_0)
FT_Psi1_pos = FT_Psi0_pos; FT_Psi1_0 = FT_Psi0_0; FT_Psi1_neg = FT_Psi0_neg; FT_eta1_0 = FT_eta0_0

abs_Psi_pos_squared = np.abs(Psi_pos)**2
abs_Psi_0_squared = np.abs(Psi_0)**2
abs_Psi_neg_squared = np.abs(Psi_neg)**2
abs_eta_0_squared = np.abs(eta_0)**2

rho = abs_Psi_pos_squared + abs_Psi_0_squared + abs_Psi_neg_squared + abs_eta_0_squared

PsiX_pos = (+ p - Vx) * Psi_pos - 2 * c * ((abs_Psi_pos_squared + abs_Psi_0_squared + abs_eta_0_squared) * Psi_pos + 0.5 * (Psi_0**2 - eta_0**2) * np.conj(Psi_neg))
PsiX_0 = ( - Vx) * Psi_0 - 2 * c * ((rho - 0.5 * abs_eta_0_squared) * Psi_0 + np.conj(Psi_0) * (0.5 * eta_0**2 + Psi_pos * Psi_neg))
PsiX_neg = (- p - Vx) * Psi_neg - 2 * c * ((abs_Psi_neg_squared + abs_Psi_0_squared + abs_eta_0_squared) * Psi_neg + 0.5 * (Psi_0**2 - eta_0**2) * np.conj(Psi_pos))
etaX_0 = (- Vx) * eta_0 - 2 * c * ((rho - 0.5 * abs_Psi_0_squared) * eta_0 + np.conj(eta_0) * (0.5 * Psi_0**2 - Psi_neg * Psi_pos))

FT_PsiX_pos = fftn(PsiX_pos); FT_PsiX_0 = fftn(PsiX_0); FT_PsiX_neg = fftn(PsiX_neg); FT_etaX_0 = fftn(etaX_0) 

#iteration parameters
dt = 0.55; c_pre = 2;               #stepsize and parameter for preconditioner      0.65 20000 1e6
Restart = 2000                     #for the condition of restarting
restarts = 0                    #for counting the number of restarts
ITER = 5e4                     #number of maximal iterations
tol=10**(-16)                   #tolerance
tol_mu = 1e-6                   # tolerance for computing mu
saving_steps = 10             #how often to save the error
abs_max_sq_11 = 100; abs_max_sq_12 = 100; abs_max_sq_21 = 100; abs_max_sq_22 = 100
jj = 0; ii = 0; i = 0 
e_11=1; e_12=1; e_21=1; e_22=1

P_inv =  (1/(c_pre - Lap))

# defining mu calculator functions
def calc_mu_11():
    mu_11 = - np.sum(np.conj(Lap * FT_Psi1_pos + FT_PsiX_pos) * P_inv * (FT_Psi1_pos)) / np.sum(np.conj(FT_Psi1_pos) * P_inv *  FT_Psi1_pos)
    return np.real(mu_11)

def calc_mu_12():
    mu_12 = - np.sum((FT_Psi1_0) * P_inv * np.conj(Lap * FT_Psi1_0 + FT_PsiX_0)) / np.sum(np.conj(FT_Psi1_0) * P_inv *  FT_Psi1_0)
    return np.real(mu_12)

def calc_mu_21():
    mu_21 = - np.sum((FT_Psi1_neg) * P_inv * np.conj(Lap * FT_Psi1_neg + FT_PsiX_neg)) / np.sum(np.conj(FT_Psi1_neg) * P_inv *  FT_Psi1_neg)
    return np.real(mu_21)

def calc_mu_22():
    mu_22 = - np.sum((FT_eta1_0) * P_inv * np.conj(Lap * FT_eta1_0 + FT_etaX_0)) / np.sum(np.conj(FT_eta1_0) * P_inv *  FT_eta1_0)
    return np.real(mu_22)

while np.max([e_11, e_12, e_21, e_22])>tol and i < ITER:
    i += 1; ii += 1; jj += 1

    # calculating mu, but only with components bigger than tol_mu
    # creating dictionary with all important information
    mu_fns = {
        "11": (abs_max_sq_11, calc_mu_11, 1),
        "12": (abs_max_sq_12, calc_mu_12, 1),
        "21": (abs_max_sq_21, calc_mu_21, 1),
        "22": (abs_max_sq_22, calc_mu_22, 1),
    }

    mu_list = []
    weights = []

    # only choosing components with abs_max > tol_mu
    for label, (value, fn, weight) in mu_fns.items():
        if value > tol_mu:
            mu_val = fn()
            mu_list.append(mu_val)
            weights.append(weight)

    if len(mu_list) == 0:
        print(f"All components smaller than {tol_mu}")  # if all components are smaller than tol_mu
        mu = None
    else:
        mu = np.average(mu_list, weights=weights)
            

    #iteration
    FT_Psi2_pos = (2 - 3/ii) * FT_Psi1_pos + dt**2 * P_inv * (Lap * FT_Psi1_pos + FT_PsiX_pos + mu*FT_Psi1_pos) - (1 - 3/ii)*FT_Psi0_pos
    Psi2_pos = ifftn(FT_Psi2_pos)

    FT_Psi2_0 = (2 - 3/ii) * FT_Psi1_0 + dt**2 * P_inv * (Lap * FT_Psi1_0 + FT_PsiX_0 + mu*FT_Psi1_0) - (1 - 3/ii)*FT_Psi0_0
    Psi2_0 = ifftn(FT_Psi2_0)

    FT_Psi2_neg = (2 - 3/ii) * FT_Psi1_neg + dt**2 * P_inv * (Lap * FT_Psi1_neg + FT_PsiX_neg + mu*FT_Psi1_neg) - (1 - 3/ii)*FT_Psi0_neg
    Psi2_neg = ifftn(FT_Psi2_neg)

    FT_eta2_0 = (2 - 3/ii) * FT_eta1_0 + dt**2 * P_inv * (Lap * FT_eta1_0 + FT_etaX_0 + mu*FT_eta1_0) - (1 - 3/ii)*FT_eta0_0
    eta2_0 = ifftn(FT_eta2_0)

    #normalize
    amp = np.sqrt( N_par / (np.sum(np.abs(Psi2_pos)**2 + np.abs(Psi2_0)**2 + np.abs(Psi2_neg)**2 + np.abs(eta2_0)**2)))
    Psi2_pos *= amp; Psi2_0 *= amp; Psi2_neg *= amp; eta2_0 *= amp
    FT_Psi2_pos *= amp; FT_Psi2_0 *= amp; FT_Psi2_neg *= amp; FT_eta2_0 *= amp

    # calculating the new squares
    abs_Psi2_pos_squared = np.abs(Psi2_pos)**2
    abs_Psi2_0_squared = np.abs(Psi2_0)**2
    abs_Psi2_neg_squared = np.abs(Psi2_neg)**2
    abs_eta2_0_squared = np.abs(eta2_0)**2

    #gradient restart
    sum1 = np.sum((np.conj(Lap * FT_Psi1_pos + FT_PsiX_pos + mu * FT_Psi1_pos)) * (FT_Psi2_pos - FT_Psi1_pos))
    sum2 = np.sum((np.conj(Lap * FT_Psi1_0 + FT_PsiX_0 + mu * FT_Psi1_0)) * (FT_Psi2_0 - FT_Psi1_0))
    sum3 = np.sum((np.conj(Lap * FT_Psi1_neg + FT_PsiX_neg + mu * FT_Psi1_neg)) * (FT_Psi2_neg - FT_Psi1_neg))
    sum4 = np.sum((np.conj(Lap * FT_eta1_0 + FT_etaX_0 + mu * FT_eta1_0)) * (FT_eta2_0 - FT_eta1_0))

    cond1 = sum1 + sum2 + sum3 + sum4
    if cond1 > 0 and ii > Restart:
        ii = 1
        restarts += 1

    rho = abs_Psi2_pos_squared + abs_Psi2_0_squared + abs_Psi2_neg_squared + abs_eta2_0_squared

    # Updating the PsiX terms
    PsiX_pos = (+ p - Vx) * Psi2_pos - 2 * c * ((abs_Psi2_pos_squared + abs_Psi2_0_squared + abs_eta2_0_squared) * Psi2_pos + 0.5 * (Psi2_0**2 - eta2_0**2) * np.conj(Psi2_neg))
    PsiX_0 = ( - Vx) * Psi2_0 - 2 * c * ((rho - 0.5 * abs_eta2_0_squared) * Psi2_0 + np.conj(Psi2_0) * (0.5 * eta2_0**2 + Psi2_pos * Psi2_neg))
    PsiX_neg = (- p - Vx) * Psi2_neg - 2 * c * ((abs_Psi2_neg_squared + abs_Psi2_0_squared + abs_eta2_0_squared) * Psi2_neg + 0.5 * (Psi2_0**2 - eta2_0**2) * np.conj(Psi2_pos))
    etaX_0 = (- Vx) * eta2_0 - 2 * c * ((rho - 0.5 * abs_Psi2_0_squared) * eta2_0 + np.conj(eta2_0) * (0.5 * Psi2_0**2 - Psi2_neg * Psi2_pos))

    FT_PsiX_pos = fftn(PsiX_pos); FT_PsiX_0 = fftn(PsiX_0); FT_PsiX_neg = fftn(PsiX_neg); FT_etaX_0 = fftn(etaX_0) 

    # calculating the error
    if i % saving_steps == 0:
        e_11 = np.sqrt((1/N_points) * np.sum(np.abs(Lap * FT_Psi2_pos + FT_PsiX_pos + mu*FT_Psi2_pos)**2))
        e_12 = np.sqrt((1/N_points) * np.sum(np.abs(Lap * FT_Psi2_0 + FT_PsiX_0 + mu*FT_Psi2_0)**2))
        e_21 = np.sqrt((1/N_points) * np.sum(np.abs(Lap * FT_Psi2_neg + FT_PsiX_neg + mu*FT_Psi2_neg)**2))
        e_22 = np.sqrt((1/N_points) * np.sum(np.abs(Lap * FT_eta2_0 + FT_etaX_0 + mu*FT_eta2_0)**2))

        Error_11.append(e_11)
        Error_12.append(e_12)
        Error_21.append(e_21)
        Error_22.append(e_22)

    # updating the maximas
    abs_max_sq_11 = np.max(np.abs(Psi2_pos)**2); abs_max_sq_12 = np.max(np.abs(Psi2_0)**2)
    abs_max_sq_21 = np.max(np.abs(Psi2_neg)**2); abs_max_sq_22 = np.max(np.abs(eta2_0)**2)

    # updating wavefunctions
    FT_Psi0_pos = FT_Psi1_pos; FT_Psi1_pos = FT_Psi2_pos
    FT_Psi0_0 = FT_Psi1_0; FT_Psi1_0 = FT_Psi2_0
    FT_Psi0_neg = FT_Psi1_neg; FT_Psi1_neg = FT_Psi2_neg
    FT_eta0_0 = FT_eta1_0; FT_eta1_0 = FT_eta2_0

    if i % 20000 == 0:
        print("At timestep:", i)

Iterations = i
x_Iter = np.arange(0, Iterations, saving_steps)

# defining the matrix components
Psi_11 = Psi2_pos; Psi_22 = Psi2_neg; Psi_12 = (1 / np.sqrt(2)) * (Psi2_0 + eta2_0); Psi_21 = (1 / np.sqrt(2)) * (Psi2_0 - eta2_0) 

#calculating spin components like on the poster
F_x = (1 / np.sqrt(2)) * (np.conj(Psi_pos) * (Psi_0 + eta_0) + np.conj(Psi_0) * (Psi_pos + Psi_neg) + np.conj(Psi_neg) * (Psi_0 + eta_0) + np.conj(eta_0) * (Psi_neg + Psi_neg))
F_y = (1j / np.sqrt(2)) *(np.conj(Psi_pos) * (Psi_0 + eta_0) + np.conj(Psi_0) * (- Psi_pos + Psi_neg) + np.conj(Psi_neg) * (- Psi_0 + eta_0) - np.conj(eta_0) * (Psi_neg + Psi_neg))
F_z = np.abs(Psi_pos)**2 - np.conj(Psi_0)*eta_0 - np.conj(eta_0) * Psi_0 - np.abs(Psi_neg)**2
Magnetisation_per_N_real = np.sum(np.abs(Psi2_pos)**2 - np.abs(Psi2_neg)**2) / N_par

F_perp = np.real(F_x + 1j * F_y)
F_z = np.real(F_z)

# calculating spin components with pauli matrices
F_x = np.conj(Psi2_pos) * Psi2_neg + np.conj(Psi2_neg) * Psi2_pos + np.conj(Psi2_0) * eta2_0 + np.conj(eta2_0) * Psi2_0
F_y = 1j * (np.conj(Psi2_neg) * Psi2_pos - np.conj(Psi2_pos) * Psi2_neg - np.conj(Psi2_0) * eta2_0 + np.conj(eta2_0) * Psi2_0)
F_z = np.conj(Psi2_pos) * Psi2_pos + np.conj(Psi2_0) * Psi2_0 - np.conj(Psi2_neg) * Psi2_neg - np.conj(eta2_0) * eta2_0
Magnetisation_per_N = np.sum(F_z) / N_par

F_perp = np.real(F_x + 1j * F_y)
F_z = np.real(F_z)

# showing interesting results
print('Number of Iterations =', Iterations)
print('Number of Resets =', restarts)
print('Number of particles = ', np.sum(rho))
print("c unitless =", c)
print("Maximal density = ", np.max(rho))
print("chemical potential mu = ", mu)
print("Magnetisation per N = ", Magnetisation_per_N_real)

# Plot of the results 
fig, ax = plt.subplots(2, 2, figsize=(15, 10)) 

# First plot
ax[0,0].plot(x, rho, color = 'grey', label = r"$\rho$")
ax[0,0].plot(x, np.abs(Psi2_pos)**2, color='green', label= r'$\Psi_{1}$')
ax[0,0].plot(x, np.abs(Psi2_0)**2, color='blue', label= r'$\Psi_{0}$', linestyle = 'dashed')
ax[0,0].plot(x, np.abs(Psi2_neg)**2, color='red', label= r'$\Psi_{-1}$', linestyle = 'dashdot')
ax[0,0].plot(x, np.abs(eta2_0)**2, color='purple', label= r'$\eta_{0}$', linestyle = 'dotted')
ax[0,0].legend()
ax[0,0].set_title(r"Groundstate in $(\Psi_1, \Psi_0, \Psi_{-1}, \eta_0)$") 
ax[0,0].set_xlabel("x")
ax[0,0].set_ylabel(rf"$|\Psi(x)|^2$")

# Second Plot
ax[0,1].plot(x_Iter, np.abs(Error_11), color='green', label=r"$\Psi_{pos}$")
ax[0,1].plot(x_Iter, np.abs(Error_12), color='blue', label=r"$\Psi_{0}$", linestyle = 'dashed')
ax[0,1].plot(x_Iter, np.abs(Error_21), color='red', label=r"$\Psi_{neg}$", linestyle = 'dashdot')
ax[0,1].plot(x_Iter, np.abs(Error_22), color='purple', label=r"$\eta_{0}$", linestyle = '--')
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

# Fourth Plot
ax[1,1].plot(x, rho, color = 'grey', label = r"$\rho$")
ax[1,1].plot(x, np.abs(Psi_11)**2, color='green', label= f'$(\\mu, p)=$ ({np.round(mu, 8)}, {np.round(p,8)}) \n' + fr'$c =$ {np.round(c, 10)}')
ax[1,1].plot(x, np.abs(Psi_12)**2, color='blue', label= r'$\Psi_{12}$', linestyle = 'dashed')
ax[1,1].plot(x, np.abs(Psi_21)**2, color='red', label= r'$\Psi_{21}$', linestyle = 'dashdot')
ax[1,1].plot(x, np.abs(Psi_22)**2, color='purple', label= r'$\Psi_{22}$', linestyle = '--')
ax[1,1].legend()
ax[1,1].set_title("Ground state") 
ax[1,1].set_xlabel("x")
ax[1,1].set_ylabel(rf"$|\Psi(x)|^2$")

plt.show()