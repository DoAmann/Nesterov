import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn 
import scipy.constants as const

def groundstates(N_par, p, params):
    # properties of the grid
    N_points = 2**11 # number of gridpoints
    #L = N_points // 4 # the unitless space
    L = 2**10
    ddx = 2 * L / N_points # spacing in the uniless space

    x = np.arange(- L, L, ddx)
    k_array = 2*np.pi*np.fft.fftfreq(N_points, ddx)  # momentum grid

    # define physical properties
    m = 1.44*10**(-25) # atom mass
    l = 220e-6 # length of the system in meters
    n = N_par / N_points # particle density
    q = 0 # units in Hz
    omega_parallel = 2* np.pi * 4 # Trapping frequency in longitudinal direction in Hz
    omega_perp =  2 * np.pi * 400 #  Trapping frequency in transverse direction in Hz (isotropic in transverse direction)
    a_B = const.physical_constants ["Bohr radius"] [0] # Bohr radius in meter
    a0 = 101.8*a_B #Scattering length 0 channel
    a2 = 100.4*a_B #Scattering length 2 channel
    dx = l  / (N_points * ddx) # distance of two points in meters 
    sigma = L / (2 * np.sqrt(2)) # std of initial gaussian guess in unitless dimensions

    # calculating other parameters
    Tscale = m * dx**2 / const.hbar
    a_HO = np.sqrt(const.hbar/(m*omega_perp)) # harmonic oscillator length of transverse trap
    c0 = 4*np.pi*const.hbar**2/(3*m)*(a0+2*a2)/(2*np.pi*a_HO**2) # density-density interactions

    # transforming in unitless dimensions
    c0 *= m *dx/ (const.hbar**2) # make unitless
    c = c0 # already unitless, spin-spin interactions
    q *= 2 * np.pi * Tscale
    p *= 2 * np.pi * Tscale
    omega_parallel *= Tscale  #making frequency unitless

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

    norm = np.sqrt( N_par / (ddx * np.sum(np.abs(Psi_i_0)**2 + np.abs(Psi_i_neg)**2 + np.abs(Psi_i_pos)**2)))
    Psi_i_0 *= norm; Psi_i_neg *= norm; Psi_i_pos *= norm 

    # initial iterate
    Psi_0 = Psi_i_0; Psi_neg = Psi_i_neg; Psi_pos = Psi_i_pos
    FT_Psi0_0 = fftn(Psi_0); FT_Psi0_neg = fftn(Psi_neg); FT_Psi0_pos = fftn(Psi_pos)
    FT_Psi1_0 = FT_Psi0_0; FT_Psi1_neg = FT_Psi0_neg; FT_Psi1_pos = FT_Psi0_pos 

    rho = np.abs(Psi_0)**2 + np.abs(Psi_neg)**2 + np.abs(Psi_pos)**2
    FT_PsiX_0 = fftn( -Vx * Psi_0 - c * rho * Psi_0) + fftn( - c * (np.abs(Psi_pos)**2 + np.abs(Psi_neg)**2) * Psi_0 - 2 * c * Psi_neg * np.conj(Psi_0) * Psi_pos)
    FT_PsiX_neg = fftn( -Vx * Psi_neg - c * rho * Psi_neg) + fftn((- p - q) * Psi_neg - c * (np.abs(Psi_neg)**2 + np.abs(Psi_0)**2 - np.abs(Psi_pos)**2) * Psi_neg - c * (Psi_0)**2 * np.conj(Psi_pos))
    FT_PsiX_pos = fftn( -Vx * Psi_pos - c * rho * Psi_pos) + fftn((p - q) * Psi_pos - c * (np.abs(Psi_pos)**2 + np.abs(Psi_0)**2 - np.abs(Psi_neg)**2) * Psi_pos - c * (Psi_0)**2 * np.conj(Psi_neg))

    #iteration parameters
    dt = params[0]; c_pre = params[1];               #stepsize and parameter for preconditioner
    Restart = params[2]                     #for the condition of restarting
    restarts = 0                    #for counting the number of restarts
    ITER = 100000                     #number of maximal iterations
    tol=10**(-11)                   #tolerance
    tol_mu = 1e-6                      # choose how mu is calculated
    max_0 = 100; max_neg = 100; max_pos = 100
    jj = 0; ii = 0; i = 0 
    e_0=1; e_neg=1; e_pos=1

    P_inv =  (1/(c_pre - Lap))

    q_val = np.copy(q)
    while np.max([e_0, e_neg, e_pos])>tol and i < ITER:
        i += 1; ii += 1; jj += 1

        # for small q
        if (q_val != 0): 
            if np.abs(q_val) < 0.01 and i < 1000:
                q = 0.01
            elif np.abs(q_val) < 0.01 and 1000 < i < 1500:
                q = (q_val + q)/2
            elif np.abs(q_val) < 0.01 and i > 1500:
                q = q_val
        elif (q_val != 0) and (q_val < 0): 
            if np.abs(q_val) < 0.01 and i < 1000:
                q = -0.01
            elif np.abs(q_val) < 0.01 and 1000 < i < 1500:
                q = (q_val + q)/2
            elif np.abs(q_val) < 0.01 and i > 1500:
                q = q_val

        # calculate mu
        if max_0 > tol_mu:
            if (max_neg > tol_mu) and (max_pos > tol_mu):
                mu_0 = -np.sum(np.conj( (1/2) * Lap * FT_Psi1_0 + FT_PsiX_0) * P_inv * FT_Psi1_0) / np.sum(np.conj(FT_Psi1_0) * P_inv * FT_Psi1_0)
                mu_0 = np.real(mu_0)

                mu_neg = -np.sum(np.conj( (1/2) * Lap * FT_Psi1_neg + FT_PsiX_neg) * P_inv * FT_Psi1_neg) / np.sum(np.conj(FT_Psi1_neg) * P_inv * FT_Psi1_neg)
                mu_neg = np.real(mu_neg)

                mu_pos = -np.sum(np.conj( (1/2) * Lap * FT_Psi1_pos + FT_PsiX_pos) * P_inv * FT_Psi1_pos) / np.sum(np.conj(FT_Psi1_pos) * P_inv * FT_Psi1_pos)
                mu_pos = np.real(mu_pos)

                mu = (2 * mu_0 + mu_neg + mu_pos)/4

            elif (max_neg < tol_mu) and (max_pos > tol_mu):
                mu_0 = -np.sum(np.conj( (1/2) * Lap * FT_Psi1_0 + FT_PsiX_0) * P_inv * FT_Psi1_0) / np.sum(np.conj(FT_Psi1_0) * P_inv * FT_Psi1_0)
                mu_0 = np.real(mu_0)

                mu_pos = -np.sum(np.conj( (1/2) * Lap * FT_Psi1_pos + FT_PsiX_pos) * P_inv * FT_Psi1_pos) / np.sum(np.conj(FT_Psi1_pos) * P_inv * FT_Psi1_pos)
                mu_pos = np.real(mu_pos)

                mu = (mu_0 + mu_pos)/2

            elif (max_neg > tol_mu) and (max_pos < tol_mu):
                mu_0 = -np.sum(np.conj( (1/2) * Lap * FT_Psi1_0 + FT_PsiX_0) * P_inv * FT_Psi1_0) / np.sum(np.conj(FT_Psi1_0) * P_inv * FT_Psi1_0)
                mu_0 = np.real(mu_0)

                mu_neg = -np.sum(np.conj( (1/2) * Lap * FT_Psi1_neg + FT_PsiX_neg) * P_inv * FT_Psi1_neg) / np.sum(np.conj(FT_Psi1_neg) * P_inv * FT_Psi1_neg)
                mu_neg = np.real(mu_neg)

                mu = (mu_0 + mu_neg)/2
            
            elif (max_neg < tol_mu) and (max_pos < tol_mu):
                mu_0 = -np.sum(np.conj( (1/2) * Lap * FT_Psi1_0 + FT_PsiX_0) * P_inv * FT_Psi1_0) / np.sum(np.conj(FT_Psi1_0) * P_inv * FT_Psi1_0)
                mu = np.real(mu_0)
        
        else:
            if (max_neg > tol_mu) and (max_pos > tol_mu):
                mu_neg = -np.sum(np.conj( (1/2) * Lap * FT_Psi1_neg + FT_PsiX_neg) * P_inv * FT_Psi1_neg) / np.sum(np.conj(FT_Psi1_neg) * P_inv * FT_Psi1_neg)
                mu_neg = np.real(mu_neg)

                mu_pos = -np.sum(np.conj( (1/2) * Lap * FT_Psi1_pos + FT_PsiX_pos) * P_inv * FT_Psi1_pos) / np.sum(np.conj(FT_Psi1_pos) * P_inv * FT_Psi1_pos)
                mu_pos = np.real(mu_pos)

                mu = (mu_neg + mu_pos)/2

            elif (max_neg < tol_mu) and (max_pos > tol_mu):
                mu_pos = -np.sum(np.conj( (1/2) * Lap * FT_Psi1_pos + FT_PsiX_pos) * P_inv * FT_Psi1_pos) / np.sum(np.conj(FT_Psi1_pos) * P_inv * FT_Psi1_pos)
                mu = np.real(mu_pos)

            elif (max_neg > tol_mu) and (max_pos < tol_mu):
                mu_neg = -np.sum(np.conj( (1/2) * Lap * FT_Psi1_neg + FT_PsiX_neg) * P_inv * FT_Psi1_neg) / np.sum(np.conj(FT_Psi1_neg) * P_inv * FT_Psi1_neg)
                mu = np.real(mu_neg)
            
            else:
                print("Error: All components are close to zero")

        #iteration
        FT_Psi2_0 = (2 - 3/ii) * FT_Psi1_0 + dt**2 * P_inv * ((1/2) * Lap * FT_Psi1_0 + FT_PsiX_0 + mu*FT_Psi1_0) - (1 - 3/ii)*FT_Psi0_0
        Psi2_0 = ifftn(FT_Psi2_0)

        FT_Psi2_neg = (2 - 3/ii) * FT_Psi1_neg + dt**2 * P_inv * ((1/2) * Lap * FT_Psi1_neg + FT_PsiX_neg + mu*FT_Psi1_neg) - (1 - 3/ii)*FT_Psi0_neg
        Psi2_neg = ifftn(FT_Psi2_neg)

        FT_Psi2_pos = (2 - 3/ii) * FT_Psi1_pos + dt**2 * P_inv * ((1/2) * Lap * FT_Psi1_pos + FT_PsiX_pos + mu*FT_Psi1_pos) - (1 - 3/ii)*FT_Psi0_pos
        Psi2_pos = ifftn(FT_Psi2_pos)

        #normalize
        amp = np.sqrt( N_par / (ddx * np.sum(np.abs(Psi2_0)**2 + np.abs(Psi2_neg)**2 + np.abs(Psi2_pos)**2)))
        Psi2_0 *= amp; Psi2_neg *= amp; Psi2_pos *= amp
        FT_Psi2_0 *= amp; FT_Psi2_neg *= amp; FT_Psi2_pos *= amp

        #gradient restart
        sum1 = np.sum((np.conj((1/2) * Lap * FT_Psi1_0 + FT_PsiX_0 + mu * FT_Psi1_0)) * (FT_PsiX_0 - FT_Psi1_0))
        sum2 = np.sum((np.conj((1/2) * Lap * FT_Psi1_neg + FT_PsiX_neg + mu * FT_Psi1_neg)) * (FT_PsiX_neg - FT_Psi1_neg))
        sum3 = np.sum((np.conj((1/2) * Lap * FT_Psi1_pos + FT_PsiX_pos + mu * FT_Psi1_pos)) * (FT_PsiX_pos - FT_Psi1_pos))

        cond1 = sum1 + sum2 + sum3
        if cond1 > 0 and ii > Restart:
            ii = 1
            restarts += 1

        rho = np.abs(Psi2_0)**2 + np.abs(Psi2_neg)**2 + np.abs(Psi2_pos)**2 

        # calculating residual error
        FT_PsiX_0 = fftn( -Vx * Psi2_0 - c * rho * Psi2_0) + fftn( - c * (np.abs(Psi2_pos)**2 + np.abs(Psi2_neg)**2) * Psi2_0 - 2 * c * Psi2_neg * np.conj(Psi2_0) * Psi2_pos)
        FT_PsiX_neg = fftn( -Vx * Psi2_neg - c * rho * Psi2_neg) + fftn( (- p - q) * Psi2_neg - c * (np.abs(Psi2_neg)**2 + np.abs(Psi2_0)**2 - np.abs(Psi2_pos)**2) * Psi2_neg - c * (Psi2_0)**2 * np.conj(Psi2_pos))
        FT_PsiX_pos = fftn( -Vx * Psi2_pos - c * rho * Psi2_pos) + fftn( (p - q) * Psi2_pos - c * (np.abs(Psi2_pos)**2 + np.abs(Psi2_0)**2 - np.abs(Psi2_neg)**2) * Psi2_pos - c * (Psi2_0)**2 * np.conj(Psi2_neg))

        e_0 = np.sqrt( (ddx / N_points) * np.sum(np.abs((1/2) * Lap * FT_Psi2_0 + FT_PsiX_0 + mu * FT_Psi2_0)**2))
        e_neg = np.sqrt( (ddx / N_points) * np.sum(np.abs((1/2) * Lap * FT_Psi2_neg + FT_PsiX_neg + mu * FT_Psi2_neg)**2))
        e_pos = np.sqrt( (ddx / N_points) * np.sum(np.abs((1/2) * Lap * FT_Psi2_pos + FT_PsiX_pos + mu * FT_Psi2_pos)**2))

        Error_0.append(e_0)
        Error_neg.append(e_neg)
        Error_pos.append(e_pos)

        # redefine the maximas
        max_0 = np.max(np.abs(Psi2_0)**2); max_neg = np.max(np.abs(Psi2_neg)**2); max_pos = np.max(np.abs(Psi2_pos)**2)

        # not stopping before q = q_val
        if q != q_val:
            e_0 = 1; e_neg = 1; e_pos = 1

        # updating wavefunctions
        FT_Psi0_0 = FT_Psi1_0; FT_Psi1_0 = FT_Psi2_0
        FT_Psi0_neg = FT_Psi1_neg; FT_Psi1_neg = FT_Psi2_neg
        FT_Psi0_pos = FT_Psi1_pos; FT_Psi1_pos = FT_Psi2_pos


    Iterations = i
    x_Iter = np.arange(0, Iterations, 1)

    # checking if q is corret
    if q != q_val:
        print("q is not ", q_val)

    # getting the physical wavefunction
    Psi2_0_phys =  Psi2_0 / np.sqrt(dx)
    Psi2_neg_phys = Psi2_neg / np.sqrt(dx)
    Psi2_pos_phys = Psi2_pos / np.sqrt(dx)

    #calculating the spin-components
    F_perp = np.sqrt(2) * (np.conj(Psi2_0) * Psi2_pos + np.conj(Psi2_neg) * Psi2_0)
    F_z = np.abs(Psi2_pos)**2 - np.abs(Psi2_neg)**2

    max_F_perp = np.max(np.abs(F_perp))
    max_F_z = np.max(np.abs(F_z))

    print("Number of Iterations = ", Iterations)

    return Psi2_0, Psi2_neg, Psi2_pos, F_perp, F_z, mu, p

# collecting mu values for different particle numbers
p = 0
N_par = np.linspace(1, 10, 40)
params = [0.7, 2, 6000]
mu_array = []

for particles in N_par:
    mu = groundstates(particles, p, params)[5]
    mu_array.append(mu)

mu_values = np.array(mu_array)

np.savez("for_csquared.npz", array1 = N_par, array2 = mu_values)