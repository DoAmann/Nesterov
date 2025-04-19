from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn 
import scipy.constants as const
import multiprocessing as mp
import time

def free_BEC_groundstate(mu_zeeman, p, params):
    # properties of the grid
    N_points = 2**11 # number of gridpoints

    x = np.arange(- N_points // 2, N_points // 2, 1)
    k_array = 2*np.pi*np.fft.fftfreq(N_points, 1)  # momentum grid

    # define physical properties
    N_par = 2e0 # number of particles
    m = 1.44*10**(-25) # atom mass
    l = 220e-6 # length of the system in meters
    n = N_par / N_points # particle density
    q = 0.0 # quadratic Zeeman shift units in Hz
    omega_parallel = 2* np.pi * 4 # Trapping frequency in longitudinal direction in Hz
    omega_perp =  2 * np.pi * 400 #  Trapping frequency in transverse direction in Hz (isotropic in transverse direction)
    a_B = const.physical_constants ["Bohr radius"] [0] # Bohr radius in meter
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
    c = c0 # already unitless, spin-spin interactions
    q *= 2 * np.pi * Tscale # making quadratic Zeeman shift dimensionless
    p *= 2 * np.pi * Tscale # making linear Zeeman shift dimensionless
    omega_parallel *= Tscale  #making frequency unitless
    mu_zeeman *= Tscale / const.hbar # making chemical potential dimensionless 

    # define potential
    def V(x):
        return 1/2 * (omega_parallel * x)**2

    Vx = V(x)
    Lap = - k_array**2

    # define lists to collect the errors

    Error_0 = []; Error_neg = []; Error_pos = []

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

    rho = np.abs(Psi_0)**2 + np.abs(Psi_neg)**2 + np.abs(Psi_pos)**2
    FT_PsiX_0 = fftn( -Vx * Psi_0 - c * rho * Psi_0 + mu_zeeman * Psi_0 - c * (np.abs(Psi_pos)**2 + np.abs(Psi_neg)**2) * Psi_0 - 2 * c * Psi_neg * np.conj(Psi_0) * Psi_pos)
    FT_PsiX_neg = fftn( -Vx * Psi_neg - c * rho * Psi_neg + (- p - q + mu_zeeman) * Psi_neg - c * (np.abs(Psi_neg)**2 + np.abs(Psi_0)**2 - np.abs(Psi_pos)**2) * Psi_neg - c * (Psi_0)**2 * np.conj(Psi_pos))
    FT_PsiX_pos = fftn( -Vx * Psi_pos - c * rho * Psi_pos + (p - q + mu_zeeman) * Psi_pos - c * (np.abs(Psi_pos)**2 + np.abs(Psi_0)**2 - np.abs(Psi_neg)**2) * Psi_pos - c * (Psi_0)**2 * np.conj(Psi_neg))

    #iteration parameters
    dt = params[0]; c_pre = params[1];               #stepsize and parameter for preconditioner
    Restart = params[2]                     #for the condition of restarting
    restarts = 0                    #for counting the number of restarts
    ITER = 180000                     #number of maximal iterations
    tol=10**(-10)                   #tolerance
    tol_mu = 1e-1                      # choose how mu is calculated
    jj = 0; ii = 0; i = 0 
    e_0=1; e_neg=1; e_pos=1

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
        sum1 = np.sum((np.conj((1/2) * Lap * FT_Psi1_0 + FT_PsiX_0)) * (FT_PsiX_0 - FT_Psi1_0))
        sum2 = np.sum((np.conj((1/2) * Lap * FT_Psi1_neg + FT_PsiX_neg)) * (FT_PsiX_neg - FT_Psi1_neg))
        sum3 = np.sum((np.conj((1/2) * Lap * FT_Psi1_pos + FT_PsiX_pos)) * (FT_PsiX_pos - FT_Psi1_pos))

        cond1 = sum1 + sum2 + sum3
        if cond1 > 0 and ii > Restart:
            ii = 1
            restarts += 1

        rho = np.abs(Psi2_0)**2 + np.abs(Psi2_neg)**2 + np.abs(Psi2_pos)**2 

        # calculating residual error
        FT_PsiX_0 = fftn( -Vx * Psi2_0 - c * rho * Psi2_0 + mu_zeeman * Psi2_0 - c * (np.abs(Psi2_pos)**2 + np.abs(Psi2_neg)**2) * Psi2_0 - 2 * c * Psi2_neg * np.conj(Psi2_0) * Psi2_pos)
        FT_PsiX_neg = fftn( -Vx * Psi2_neg - c * rho * Psi2_neg + (- p - q + mu_zeeman) * Psi2_neg - c * (np.abs(Psi2_neg)**2 + np.abs(Psi2_0)**2 - np.abs(Psi2_pos)**2) * Psi2_neg - c * (Psi2_0)**2 * np.conj(Psi2_pos))
        FT_PsiX_pos = fftn( -Vx * Psi2_pos - c * rho * Psi2_pos + (p - q + mu_zeeman) * Psi2_pos - c * (np.abs(Psi2_pos)**2 + np.abs(Psi2_0)**2 - np.abs(Psi2_neg)**2) * Psi2_pos - c * (Psi2_0)**2 * np.conj(Psi2_neg))

        e_0 = np.sqrt( (1 / N_points) * np.sum(np.abs((1/2) * Lap * FT_Psi2_0 + FT_PsiX_0)**2))
        e_neg = np.sqrt( (1 / N_points) * np.sum(np.abs((1/2) * Lap * FT_Psi2_neg + FT_PsiX_neg)**2))
        e_pos = np.sqrt( (1 / N_points) * np.sum(np.abs((1/2) * Lap * FT_Psi2_pos + FT_PsiX_pos)**2))

        Error_0.append(e_0)
        Error_neg.append(e_neg)
        Error_pos.append(e_pos)

        # updating wavefunctions
        FT_Psi0_0 = FT_Psi1_0; FT_Psi1_0 = FT_Psi2_0
        FT_Psi0_neg = FT_Psi1_neg; FT_Psi1_neg = FT_Psi2_neg
        FT_Psi0_pos = FT_Psi1_pos; FT_Psi1_pos = FT_Psi2_pos


    Iterations = i
    x_Iter = np.arange(0, Iterations, 1)

    # getting the physical wavefunction
    Psi2_0_phys =  Psi2_0 / np.sqrt(dx)
    Psi2_neg_phys = Psi2_neg / np.sqrt(dx)
    Psi2_pos_phys = Psi2_pos / np.sqrt(dx)

    # calculating number of particles
    Nparticles = np.sum(rho)

    #calculating the spin-components
    F_perp = np.sqrt(2) * (np.conj(Psi2_0) * Psi2_pos + np.conj(Psi2_neg) * Psi2_0)
    F_z = np.abs(Psi2_pos)**2 - np.abs(Psi2_neg)**2
    Magnetisation = np.sum(F_z)
    Magnetisation_per_N = Magnetisation / Nparticles

    max_F_perp = np.max(np.abs(F_perp))
    max_F_z = np.max(np.abs(F_z))

    # showing interesting results
    print('Number of Iterations =', Iterations)

    #calculating max density
    rho_max = np.max(rho)

    #calculating the energy
    if (c == 0):
        # calculating Laplace
        Lap_0 = ifftn( (1/2) * Lap * FT_Psi2_0)
        Lap_neg = ifftn( (1/2) * Lap * FT_Psi2_neg)
        Lap_pos = ifftn( (1/2) * Lap * FT_Psi2_pos)

        # calculating energy densities
        energy_0 = np.conj(Psi2_0) * Lap_0 + np.conj(Psi2_0) * (V(x) - mu_zeeman) * Psi2_0
        energy_neg = np.conj(Psi2_neg) * Lap_neg + np.conj(Psi2_neg) * (V(x) - mu_zeeman + p + q) * Psi2_neg
        energy_pos = np.conj(Psi2_pos) * Lap_pos + np.conj(Psi2_pos) * (V(x) - mu_zeeman - p + q) * Psi2_pos

        # calculating the energy
        energy_tot = np.real(energy_0 + energy_neg + energy_pos)
        Energy = np.sum(energy_tot)

    if Nparticles >= 1:
        return Psi2_0, Psi2_neg, Psi2_pos, F_perp, F_z, mu_zeeman, p, rho_max, Magnetisation, Magnetisation_per_N, Nparticles, Energy
    else:
        Psi2_0, Psi2_neg, Psi2_pos, F_perp, F_z, rho_max, Magnetisation, Magnetisation_per_N, Nparticles, Energy = (np.zeros(N_points),np.zeros(N_points),np.zeros(N_points),np.zeros(N_points),np.zeros(N_points),0,0,0,0,0)
        return Psi2_0, Psi2_neg, Psi2_pos, F_perp, F_z, mu_zeeman, p, rho_max, Magnetisation, Magnetisation_per_N, Nparticles, Energy
    
def Occupation_distribution(mu_zeeman):
    params = [0.7, 2, 4000]
    groundstate = free_BEC_groundstate(mu_zeeman, 0, params)
    N_par = groundstate[10]
    return N_par

mu_array = np.linspace(1/1.17 * 1.530982e-33, 1/1.16 * 1.530982e-33, 100)

# using pool
if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()) as pool:
        N_par_array = pool.map(Occupation_distribution, mu_array)

    # saving data
    np.savez("mu_vs_Npar_mu_code.npz", mu=mu_array, N_par=N_par_array)