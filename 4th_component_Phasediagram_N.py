import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn 
import scipy.constants as const
import multiprocessing as mp

def linear_Zeeman_groundstate(N_par: float, p: float, params):
    #properties of the grid
    N_points = 2**11 #number of gridpoints
    x = np.arange(- N_points // 2, N_points // 2, 1) # spatial grid
    k_array = 2*np.pi*np.fft.fftfreq(N_points, 1)  # momentum grid
    dk = np.abs(k_array[1] - k_array[0]) # spacing in momentum space

    # define physical properties
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

    # define potential
    def V(x):
        return (omega_parallel * x)**2

    Vx = V(x) 
    Lap = - k_array**2

    # define lists to collect the errors
    Error_11 = []; Error_12 = []; Error_21 = []; Error_22 = []

    # define initial wavefunction
    def Psi_Initial_11(x):
        return np.exp(- x**2 / (2 * sigma**2)) * (1 + 0.05 * np.random.random(N_points))

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

    PsiX_11 = (+ p - Vx) * Psi_11 - 2 * c * ((abs_Psi_11_squared + abs_Psi_21_squared + abs_Psi_12_squared) * Psi_11 + Psi_12 * np.conj(Psi_22) * Psi_21)
    PsiX_12 = ( - Vx) * Psi_12 - 2 * c * ((abs_Psi_11_squared + abs_Psi_22_squared + abs_Psi_12_squared) * Psi_12 + Psi_11 * np.conj(Psi_21) * Psi_22)
    PsiX_21 = ( - Vx) * Psi_21 - 2 * c * ((abs_Psi_11_squared + abs_Psi_21_squared + abs_Psi_22_squared) * Psi_21 + Psi_22 * np.conj(Psi_12) * Psi_11)
    PsiX_22 = (- p - Vx) * Psi_22 - 2 * c * ((abs_Psi_21_squared + abs_Psi_12_squared + abs_Psi_22_squared) * Psi_22 + Psi_21 * np.conj(Psi_11) * Psi_12)

    FT_PsiX_11 = fftn(PsiX_11); FT_PsiX_12 = fftn(PsiX_12); FT_PsiX_21 = fftn(PsiX_21); FT_PsiX_22 = fftn(PsiX_22) 

    #iteration parameters
    dt = params[0]; c_pre = params[1];               #stepsize and parameter for preconditioner
    Restart = params[2]                     #for the condition of restarting                    #for the condition of restarting
    restarts = 0                    #for counting the number of restarts
    ITER = 90000                     #number of maximal iterations
    tol=10**(-12)                   #tolerance
    tol_mu = 1e-4                   # tolerance for computing mu
    abs_max_sq_11 = 100; abs_max_sq_12 = 100; abs_max_sq_21 = 100; abs_max_sq_22 = 100
    jj = 0; ii = 0; i = 0 
    e_11=1; e_12=1; e_21=1; e_22=1

    P_inv =  (1/(c_pre - Lap))

    # defining mu calculator functions
    def calc_mu_11():
        mu_11 = - np.sum(np.conj(Lap * FT_Psi1_11 + FT_PsiX_11) * P_inv * (FT_Psi1_11)) / np.sum(np.conj(FT_Psi1_11) * P_inv *  FT_Psi1_11)
        return np.real(mu_11)

    def calc_mu_12():
        mu_12 = - np.sum((FT_Psi1_12) * P_inv * np.conj(Lap * FT_Psi1_12 + FT_PsiX_12)) / np.sum(np.conj(FT_Psi1_12) * P_inv *  FT_Psi1_12)
        return np.real(mu_12)

    def calc_mu_21():
        mu_21 = - np.sum((FT_Psi1_21) * P_inv * np.conj(Lap * FT_Psi1_21 + FT_PsiX_21)) / np.sum(np.conj(FT_Psi1_21) * P_inv *  FT_Psi1_21)
        return np.real(mu_21)

    def calc_mu_22():
        mu_22 = - np.sum((FT_Psi1_22) * P_inv * np.conj(Lap * FT_Psi1_22 + FT_PsiX_22)) / np.sum(np.conj(FT_Psi1_22) * P_inv *  FT_Psi1_22)
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
        FT_Psi2_11 = (2 - 3/ii) * FT_Psi1_11 + dt**2 * P_inv * (Lap * FT_Psi1_11 + FT_PsiX_11 + mu*FT_Psi1_11) - (1 - 3/ii)*FT_Psi0_11
        Psi2_11 = ifftn(FT_Psi2_11)

        FT_Psi2_12 = (2 - 3/ii) * FT_Psi1_12 + dt**2 * P_inv * (Lap * FT_Psi1_12 + FT_PsiX_12 + mu*FT_Psi1_12) - (1 - 3/ii)*FT_Psi0_12
        Psi2_12 = ifftn(FT_Psi2_12)

        FT_Psi2_21 = (2 - 3/ii) * FT_Psi1_21 + dt**2 * P_inv * (Lap * FT_Psi1_21 + FT_PsiX_21 + mu*FT_Psi1_21) - (1 - 3/ii)*FT_Psi0_21
        Psi2_21 = ifftn(FT_Psi2_21)

        FT_Psi2_22 = (2 - 3/ii) * FT_Psi1_22 + dt**2 * P_inv * (Lap * FT_Psi1_22 + FT_PsiX_22 + mu*FT_Psi1_22) - (1 - 3/ii)*FT_Psi0_22
        Psi2_22 = ifftn(FT_Psi2_22)

        #normalize
        amp = np.sqrt( N_par / (np.sum(np.abs(Psi2_11)**2 + np.abs(Psi2_12)**2 + np.abs(Psi2_21)**2 + np.abs(Psi2_22)**2)))
        Psi2_11 *= amp; Psi2_12 *= amp; Psi2_21 *= amp; Psi2_22 *= amp
        FT_Psi2_11 *= amp; FT_Psi2_12 *= amp; FT_Psi2_21 *= amp; FT_Psi2_22 *= amp

        # calculating the new squares
        abs_Psi2_11_squared = np.abs(Psi2_11)**2
        abs_Psi2_12_squared = np.abs(Psi2_12)**2
        abs_Psi2_21_squared = np.abs(Psi2_21)**2
        abs_Psi2_22_squared = np.abs(Psi2_22)**2

        #gradient restart
        sum1 = np.sum((np.conj(Lap * FT_Psi1_11 + FT_PsiX_11 + mu * FT_Psi1_11)) * (FT_Psi2_11 - FT_Psi1_11))
        sum2 = np.sum((np.conj(Lap * FT_Psi1_12 + FT_PsiX_12 + mu * FT_Psi1_12)) * (FT_Psi2_12 - FT_Psi1_12))
        sum3 = np.sum((np.conj(Lap * FT_Psi1_21 + FT_PsiX_21 + mu * FT_Psi1_21)) * (FT_Psi2_21 - FT_Psi1_21))
        sum4 = np.sum((np.conj(Lap * FT_Psi1_22 + FT_PsiX_22 + mu * FT_Psi1_22)) * (FT_Psi2_22 - FT_Psi1_22))

        cond1 = sum1 + sum2 + sum3 + sum4
        if cond1 > 0 and ii > Restart:
            ii = 1
            restarts += 1

        rho = abs_Psi2_11_squared + abs_Psi2_12_squared + abs_Psi2_21_squared + abs_Psi2_22_squared

        # Updating the PsiX terms
        PsiX_11 = (+ p - Vx) * Psi2_11 - 2 * c * ((abs_Psi2_11_squared + abs_Psi2_21_squared + abs_Psi2_12_squared) * Psi2_11 + Psi2_12 * np.conj(Psi2_22) * Psi2_21)
        PsiX_12 = ( - Vx) * Psi2_12 - 2 * c * ((abs_Psi2_11_squared + abs_Psi2_22_squared + abs_Psi2_12_squared) * Psi2_12 + Psi2_11 * np.conj(Psi2_21) * Psi2_22)
        PsiX_21 = ( - Vx) * Psi2_21 - 2 * c * ((abs_Psi2_11_squared + abs_Psi2_21_squared + abs_Psi2_22_squared) * Psi2_21 + Psi2_22 * np.conj(Psi2_12) * Psi2_11)
        PsiX_22 = (- p - Vx) * Psi2_22 - 2 * c * ((abs_Psi2_21_squared + abs_Psi2_12_squared + abs_Psi2_22_squared) * Psi2_22 + Psi2_21 * np.conj(Psi2_11) * Psi2_12)

        FT_PsiX_11 = fftn(PsiX_11); FT_PsiX_12 = fftn(PsiX_12); FT_PsiX_21 = fftn(PsiX_21); FT_PsiX_22 = fftn(PsiX_22)

        # calculating the error
        e_11 = np.sqrt((1/N_points) * np.sum(np.abs(Lap * FT_Psi2_11 + FT_PsiX_11 + mu*FT_Psi2_11)**2))
        e_12 = np.sqrt((1/N_points) * np.sum(np.abs(Lap * FT_Psi2_12 + FT_PsiX_12 + mu*FT_Psi2_12)**2))
        e_21 = np.sqrt((1/N_points) * np.sum(np.abs(Lap * FT_Psi2_21 + FT_PsiX_21 + mu*FT_Psi2_21)**2))
        e_22 = np.sqrt((1/N_points) * np.sum(np.abs(Lap * FT_Psi2_22 + FT_PsiX_22 + mu*FT_Psi2_22)**2))

        Error_11.append(e_11)
        Error_12.append(e_12)
        Error_21.append(e_21)
        Error_22.append(e_22)

        # updating the maximas
        abs_max_sq_11 = np.max(np.abs(Psi2_11)**2); abs_max_sq_12 = np.max(np.abs(Psi2_12)**2)
        abs_max_sq_21 = np.max(np.abs(Psi2_21)**2); abs_max_sq_22 = np.max(np.abs(Psi2_22)**2)

        # updating wavefunctions
        FT_Psi0_11 = FT_Psi1_11; FT_Psi1_11 = FT_Psi2_11
        FT_Psi0_12 = FT_Psi1_12; FT_Psi1_12 = FT_Psi2_12
        FT_Psi0_21 = FT_Psi1_21; FT_Psi1_21 = FT_Psi2_21
        FT_Psi0_22 = FT_Psi1_22; FT_Psi1_22 = FT_Psi2_22

    Iterations = i
    x_Iter = np.arange(0, Iterations, 1)

    # calculating the Magnetisation per particle
    Magnetisation_per_N = np.sum(np.abs(Psi2_11)**2 - np.abs(Psi2_22)**2) / N_par

    # showing interesting results
    print('Number of Iterations =', Iterations)

    #calculating max density
    rho_max = np.max(rho)

    return p, Magnetisation_per_N, N_par

# Wrapper-Funktion für multiprocessing
def run_phase_combination(args):
    N_val, p_val = args
    params = [0.55, 4, 2000]
    return linear_Zeeman_groundstate(N_val, p_val, params)


if __name__ == "__main__":
    # special chemical potential
    c_spec = 0.0005264653090365891

    # define parameter in physical units
    N_par = np.linspace(1.0, 100, 5)
    p = np.linspace(0, 5*c_spec, 10)

    all_args = [(N_val, p_val) for N_val in N_par for p_val in p]

    # Parallel berechnen
    with mp.Pool(processes=mp.cpu_count()-2) as pool:
        results = pool.map(run_phase_combination, all_args)

    # Ergebnisse in Arrays umwandeln
    len_p = len(p)
    N_unitless = []
    p_unitless = []
    result_Mag_per_N = []

    for i, N_val in enumerate(N_par):
        array_Mag_per_N = []
        for j, p_val in enumerate(p):
            idx = i * len_p + j
            p_unless, Magnetisation_per_N, N_unless = results[idx]
            array_Mag_per_N.append(Magnetisation_per_N)
            if i == 0:
                p_unitless.append(p_unless)
        N_unitless.append(N_unless)
        result_Mag_per_N.append(array_Mag_per_N)

    np.savez("4th_component_Phasediagram_N.npz", 
             N_particles_array = N_unitless, 
             p_array = p_unitless,  
             Mag_per_N_array = result_Mag_per_N)