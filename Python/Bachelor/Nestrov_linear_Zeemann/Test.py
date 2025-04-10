import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn 
import scipy.constants as const
import pickle

#please insert q in Hz
def groundstate(mu_zeeman: float , p: float, params: list):
    # properties of the grid
    N_points = 2**11 # number of gridpoints

    x = np.arange(- N_points // 2, N_points // 2, 1)
    k_array = 2*np.pi*np.fft.fftfreq(N_points, 1)  # momentum grid

    # define physical properties
    N_par: int = 2e0 # number of particles
    m: float = 1.44*10**(-25) # atom mass
    l: float = 220e-6 # length of the system in meters
    n = N_par / N_points # particle density
    q = 0 # quadratic Zeeman shift units in Hz
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
    c = c0 # already unitless, spin-spin interactions
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
    Lap = - np.pow(k_array, 2)


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
    PsiX_0 = -Vx * Psi_0 - c * rho * Psi_0 + mu_zeeman * Psi_0  
    PsiX_0 += - c * (abs_Psipos_sq + abs_Psineg_sq) * Psi_0
    PsiX_0 += - 2 * c * Psi_neg * np.conj(Psi_0) * Psi_pos
        
    PsiX_neg = -Vx * Psi_neg - c * rho * Psi_neg
    PsiX_neg += (- p - q + mu_zeeman) * Psi_neg 
    PsiX_neg += - c * (abs_Psineg_sq + abs_Psi0_sq - abs_Psipos_sq) * Psi_neg
    PsiX_neg += - c * (Psi_0)**2 * np.conj(Psi_pos)
        
    PsiX_pos = -Vx * Psi_pos - c * rho * Psi_pos
    PsiX_pos += (p - q + mu_zeeman) * Psi_pos
    PsiX_pos += - c * (abs_Psipos_sq + abs_Psi0_sq - abs_Psineg_sq) * Psi_pos
    PsiX_pos += - c * (Psi_0)**2 * np.conj(Psi_neg)
        
    FT_PsiX_0 = fftn(PsiX_0)
    FT_PsiX_neg = fftn(PsiX_neg)
    FT_PsiX_pos = fftn(PsiX_pos)

    #iteration parameters
    dt = params[0]; c_pre = params[1];               #stepsize and parameter for preconditioner
    Restart = params[2]                     #for the condition of restarting
    restarts = 0                    #for counting the number of restarts
    ITER = 60000                     #number of maximal iterations
    tol=10**(-13)                   #tolerance
    jj = 0; ii = 0; i = 0 
    e_0=1; e_neg=1; e_pos=1

    # define lists to collect the errors
    Error_0 = np.ones(ITER + 1); Error_neg = np.ones(ITER + 1); Error_pos = np.ones(ITER + 1)

    P_inv =  (1/(c_pre - Lap))
    q_val = np.copy(q)
    while np.max([e_0, e_neg, e_pos])>tol and i < ITER:
        i += 1; ii += 1; jj += 1
            
        if q_val != 0:
            q_abs = np.abs(q_val)
            if q_abs < 0.01:
                if i < 1000:
                    q = np.sign(q_val) * 0.01
                elif i < 1500:
                    q = (q_val + q) * 0.5
                else:
                    q = q_val

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

        # not stopping before q = q_val
        if q != q_val:
            e_0 = 1; e_neg = 1; e_pos = 1

        # updating wavefunctions
        FT_Psi0_0 = FT_Psi1_0; FT_Psi1_0 = FT_Psi2_0
        FT_Psi0_neg = FT_Psi1_neg; FT_Psi1_neg = FT_Psi2_neg
        FT_Psi0_pos = FT_Psi1_pos; FT_Psi1_pos = FT_Psi2_pos

    Error_0 = Error_0[1:i + 1]
    Error_neg = Error_neg[1:i + 1]
    Error_pos = Error_pos[1:i + 1]

    Iterations = i
    x_Iter = np.arange(0, Iterations, 1)

    # checking if q is corret
    if q != q_val:
        print("q is not ", q_val)

    # getting the physical wavefunction
    Psi2_0_phys =  Psi2_0 / np.sqrt(dx)
    Psi2_neg_phys = Psi2_neg / np.sqrt(dx)
    Psi2_pos_phys = Psi2_pos / np.sqrt(dx)

    #calculating the spin-components and magnetisation
    F_perp = np.sqrt(2) * (np.conj(Psi2_0) * Psi2_pos + np.conj(Psi2_neg) * Psi2_0)
    F_z = np.abs(Psi2_pos)**2 - np.abs(Psi2_neg)**2
    Magnetisation = np.sum(F_z)
    Magnetisation_per_N = Magnetisation / np.sum(rho)

    max_F_perp = np.max(np.abs(F_perp))
    max_F_z = np.max(np.abs(F_z))

    # showing interesting results
    print('Number of Iterations =', Iterations)

    # calculating max_density
    rho_max = np.max(rho)

    # calculating number of particles
    N_particle = np.sum(rho)

    return Psi2_0, Psi2_neg, Psi2_pos, F_perp, F_z, mu_zeeman, p, rho_max, N_particle

mu = np.linspace(1.5309816e-33, 10 * 1.5309816e-33, 15)
p = 0

N_array = []
mu_array = []
for mu_value in mu:
    c = groundstate(mu_value, p, [0.7, 2, 6000])
    Nparticles, mu_unitless = c[8], c[5]
    mu_array.append(mu_unitless)
    N_array.append(Nparticles)

print(mu_array)
print(N_array)

plt.plot(mu_array, N_array)
plt.show()