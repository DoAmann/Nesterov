import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn 
import sympy as sp #just for the plot

#define variables and parameters
#define the potential
Omega = sp.Symbol('Omega')
x = sp.Symbol('x')
def V(x, Omega):
    return 0.5*Omega**2*x**2

V_str = sp.latex(V(x, Omega))

#define the spatial and Fourier space
L = 750              #length of the spatial space
m = 500*3             #number of descrete points 
dx = 2*L/m          #distance of two points          
Omega = 1e-2       #trepping strength
sigma = 500/np.sqrt(2)
q = 0.1
delta = 5e-3 


x = np.linspace(-L, L - dx, m)
k_array = 2*np.pi*np.fft.fftfreq(m, dx)  

#calculate Laplacian and the potential at the discrete points
Lap = -k_array**2
Vx = V(x, Omega)

#define list for tracing the errors
Error_0 = []
Error_neg = []
Error_pos = []

#initial normalization
def Psi_Initial_0(x):
    return np.exp(-x**2/(2 * sigma**2))

def Psi_Initial_neg(x):
    return np.exp(-x**2/(2 * sigma**2))

def Psi_Initial_pos(x):
    return 0.998*np.exp(-x**2/(2 * sigma**2))

#normalizing
N = 20000                           #normalization constant
Psi_i_0 = Psi_Initial_0(x)
Psi_i_neg = Psi_Initial_neg(x)
Psi_i_pos = Psi_Initial_pos(x)

Norm = (N/(np.sum(np.abs(Psi_i_0)**2 + np.abs(Psi_i_neg)**2 + np.abs(Psi_i_pos)**2)))**(1/2) 
Psi_i_0 = Psi_i_0 * Norm 
Psi_i_neg = Psi_i_neg * Norm
Psi_i_pos = Psi_i_pos * Norm

#initial iterate
Psi_0 = Psi_i_0; Psi_neg = Psi_i_neg; Psi_pos = Psi_i_pos
FT_Psi0_0 = fftn(Psi_0); FT_Psi0_neg = fftn(Psi_neg); FT_Psi0_pos = fftn(Psi_pos)
FT_Psi1_0 = FT_Psi0_0; FT_Psi1_neg = FT_Psi0_neg; FT_Psi1_pos = FT_Psi0_pos

rho = np.abs(Psi_0)**2 + np.abs(Psi_neg)**2 + np.abs(Psi_pos)**2
print('Rho = ', np.sum(rho))
FT_PsiX_0 = fftn(-Vx * Psi_0 - rho*Psi_0) + fftn( - delta * (np.abs(Psi_neg)**2 + np.abs(Psi_pos)**2) * Psi_0 - 2*delta * Psi_neg * np.conj(Psi_0) * Psi_pos)
FT_PsiX_neg = fftn(-Vx * Psi_neg - rho*Psi_neg) + fftn(- q * Psi_neg - delta * (np.abs(Psi_neg)**2 + np.abs(Psi_0)**2 - np.abs(Psi_pos)**2) * Psi_neg - delta * (Psi_0)**2*np.conj(Psi_pos))
FT_PsiX_pos = fftn(-Vx * Psi_pos - rho*Psi_pos) + fftn(- q * Psi_pos - delta * (np.abs(Psi_pos)**2 + np.abs(Psi_0)**2 - np.abs(Psi_neg)**2) * Psi_pos - delta * (Psi_0)**2*np.conj(Psi_neg))

#iteration parameters
dt = 0.5; c = 7;               #stepsize and parameter for preconditioner
Restart = 200                     #for the condition of restarting
restarts = 0                    #for counting the number of restarts
ITER = 100000                     #number of maximal iterations
tol=10**(-10)                   #tolerance
tol_mu = 2e0
jj = 0; ii = 0; i = 0 
e_0=1; e_neg=1; e_pos=1

P_inv = (1/(c - Lap))

while np.max([e_0, e_neg, e_pos])>tol and i < ITER:
    i += 1; ii += 1; jj += 1
    
    #calculate mu
    if (np.max(np.abs(FT_Psi1_0)**2) > tol_mu):
        if (np.max(np.abs(FT_Psi1_neg)**2) > tol_mu) and (np.max(np.abs(FT_Psi1_pos)**2) > tol_mu):
            mu_0 = -np.sum(np.conj(Lap * FT_Psi1_0 + FT_PsiX_0) * P_inv * ((FT_Psi1_0))) / np.sum(np.conj(FT_Psi1_0) * P_inv * (FT_Psi1_0)) 
            mu_0 = np.real(mu_0)

            mu_neg = -np.sum(np.conj(Lap * FT_Psi1_neg + FT_PsiX_neg) * P_inv * ((FT_Psi1_neg))) / np.sum(np.conj(FT_Psi1_neg) * P_inv * (FT_Psi1_neg)) 
            mu_neg = np.real(mu_neg)

            mu_pos = -np.sum(np.conj(Lap * FT_Psi1_pos + FT_PsiX_pos) * P_inv * ((FT_Psi1_pos))) / np.sum(np.conj(FT_Psi1_pos) * P_inv * (FT_Psi1_pos)) 
            mu_pos = np.real(mu_pos)

            mu = (2*mu_0 + mu_neg + mu_pos)/4
            
        elif (np.max(np.abs(FT_Psi1_neg)**2) < tol_mu) and (np.max(np.abs(FT_Psi1_pos)**2) > tol_mu):
            mu_0 = -np.sum(np.conj(Lap * FT_Psi1_0 + FT_PsiX_0) * P_inv * ((FT_Psi1_0))) / np.sum(np.conj(FT_Psi1_0) * P_inv * (FT_Psi1_0)) 
            mu_0 = np.real(mu_0)

            mu_pos = -np.sum(np.conj(Lap * FT_Psi1_pos + FT_PsiX_pos) * P_inv * ((FT_Psi1_pos))) / np.sum(np.conj(FT_Psi1_pos) * P_inv * (FT_Psi1_pos)) 
            mu_pos = np.real(mu_pos)

            mu = (mu_0 + mu_pos)/2

        elif (np.max(np.abs(FT_Psi1_neg)**2) > tol_mu) and (np.max(np.abs(FT_Psi1_pos)**2) < tol_mu):
            mu_0 = -np.sum(np.conj(Lap * FT_Psi1_0 + FT_PsiX_0) * P_inv * ((FT_Psi1_0))) / np.sum(np.conj(FT_Psi1_0) * P_inv * (FT_Psi1_0)) 
            mu_0 = np.real(mu_0)

            mu_neg = -np.sum(np.conj(Lap * FT_Psi1_neg + FT_PsiX_neg) * P_inv * ((FT_Psi1_neg))) / np.sum(np.conj(FT_Psi1_neg) * P_inv * (FT_Psi1_neg)) 
            mu_neg = np.real(mu_neg)

            mu = (mu_0 + mu_neg)/2

        else:
            mu_0 = -np.sum(np.conj(Lap * FT_Psi1_0 + FT_PsiX_0) * P_inv * ((FT_Psi1_0))) / np.sum(np.conj(FT_Psi1_0) * P_inv * (FT_Psi1_0)) 
            mu = np.real(mu_0)

    elif (np.max(np.abs(FT_Psi1_0)**2) < tol_mu):
        if (np.max(np.abs(FT_Psi1_neg)**2) > tol_mu) and (np.max(np.abs(FT_Psi1_pos)**2) > tol_mu):
            mu_neg = -np.sum(np.conj(Lap * FT_Psi1_neg + FT_PsiX_neg) * P_inv * ((FT_Psi1_neg))) / np.sum(np.conj(FT_Psi1_neg) * P_inv * (FT_Psi1_neg)) 
            mu_neg = np.real(mu_neg)

            mu_pos = -np.sum(np.conj(Lap * FT_Psi1_pos + FT_PsiX_pos) * P_inv * ((FT_Psi1_pos))) / np.sum(np.conj(FT_Psi1_pos) * P_inv * (FT_Psi1_pos)) 
            mu_pos = np.real(mu_pos)

            mu = (mu_neg + mu_pos)/2
        
        elif (np.max(np.abs(FT_Psi1_neg)**2) < tol_mu) and (np.max(np.abs(FT_Psi1_pos)**2) > tol_mu):
            mu = -np.sum(np.conj(Lap * FT_Psi1_pos + FT_PsiX_pos) * P_inv * ((FT_Psi1_pos))) / np.sum(np.conj(FT_Psi1_pos) * P_inv * (FT_Psi1_pos)) 
            mu = np.real(mu)

        elif (np.max(np.abs(FT_Psi1_neg)**2) > tol_mu) and (np.max(np.abs(FT_Psi1_pos)**2) < tol_mu):
            mu = -np.sum(np.conj(Lap * FT_Psi1_neg + FT_PsiX_neg) * P_inv * ((FT_Psi1_neg))) / np.sum(np.conj(FT_Psi1_neg) * P_inv * (FT_Psi1_neg)) 
            mu = np.real(mu)

        else:
            print("All components smaller than 1e-2")
    
    #iteration
    FT_Psi2_0 = (2 - 3/ii) * FT_Psi1_0 + dt**2 * P_inv * (Lap * FT_Psi1_0 + FT_PsiX_0 + mu*FT_Psi1_0) - (1 - 3/ii)*FT_Psi0_0
    Psi2_0 = ifftn(FT_Psi2_0)

    FT_Psi2_neg = (2 - 3/ii) * FT_Psi1_neg + dt**2 * P_inv * (Lap * FT_Psi1_neg + FT_PsiX_neg + mu*FT_Psi1_neg) - (1 - 3/ii)*FT_Psi0_neg
    Psi2_neg = ifftn(FT_Psi2_neg)

    FT_Psi2_pos = (2 - 3/ii) * FT_Psi1_pos + dt**2 * P_inv * (Lap * FT_Psi1_pos + FT_PsiX_pos + mu*FT_Psi1_pos) - (1 - 3/ii)*FT_Psi0_pos
    Psi2_pos = ifftn(FT_Psi2_pos)

    #normalization
    amp = (N/(np.sum((np.abs(Psi2_0)**2 + np.abs(Psi2_neg)**2 + np.abs(Psi2_pos)**2))))**(1/2)  
    Psi2_0 = Psi2_0 * amp
    Psi2_neg = Psi2_neg * amp
    Psi2_pos = Psi2_pos * amp
    FT_Psi2_0 = FT_Psi2_0 * amp
    FT_Psi2_neg = FT_Psi2_neg * amp
    FT_Psi2_pos = FT_Psi2_pos * amp

    #gradient restart
    sum1 = np.sum((Lap * FT_Psi1_0 + FT_PsiX_0 + mu * FT_Psi1_0) * np.conj(FT_PsiX_0 - FT_Psi1_0))
    sum2 = np.sum((Lap * FT_Psi1_neg + FT_PsiX_neg + mu * FT_Psi1_neg) * np.conj(FT_PsiX_neg - FT_Psi1_neg))
    sum3 = np.sum((Lap * FT_Psi1_pos + FT_PsiX_pos + mu * FT_Psi1_pos) * np.conj(FT_PsiX_pos - FT_Psi1_pos))

    cond1 = sum1 + sum2 + sum3
    if cond1 > 0 and ii > Restart:
        ii = 1
        restarts += 1

    rho = np.abs(Psi2_0)**2 + np.abs(Psi2_neg)**2 + np.abs(Psi2_pos)**2 

    #residual error
    FT_PsiX_0 = fftn(-Vx * Psi2_0 - rho * Psi2_0) + fftn( - delta * (np.abs(Psi2_neg)**2 + np.abs(Psi2_pos)**2) * Psi2_0 - 2*delta * Psi2_neg * np.conj(Psi2_0) * Psi2_pos)
    e_0 = np.sqrt(1/m * np.sum((FT_PsiX_0 + mu*FT_Psi2_0 + Lap * FT_Psi2_0) * np.conj(FT_PsiX_0 + mu * FT_Psi2_0 + Lap * FT_Psi2_0) ))  
    Error_0.append(e_0)

    FT_PsiX_neg = fftn(-Vx * Psi2_neg - rho * Psi2_neg) + fftn(- q * Psi2_neg - delta * (np.abs(Psi2_neg)**2 + np.abs(Psi2_0)**2 - np.abs(Psi2_pos)**2) * Psi2_neg - delta * (Psi2_0)**2*np.conj(Psi2_pos))
    e_neg = np.sqrt(1/m * np.sum((FT_PsiX_neg + mu*FT_Psi2_neg + Lap * FT_Psi2_neg) * np.conj(FT_PsiX_neg + mu * FT_Psi2_neg + Lap * FT_Psi2_neg) ))  
    Error_neg.append(e_neg)

    FT_PsiX_pos = fftn(-Vx * Psi2_pos - rho * Psi2_pos) + fftn(- q * Psi2_pos - delta * (np.abs(Psi2_pos)**2 + np.abs(Psi2_0)**2 - np.abs(Psi2_neg)**2) * Psi2_pos - delta * (Psi2_0)**2*np.conj(Psi2_neg))
    e_pos = np.sqrt(1/m * np.sum((FT_PsiX_pos + mu*FT_Psi2_pos + Lap * FT_Psi2_pos) * np.conj(FT_PsiX_pos + mu * FT_Psi2_pos + Lap * FT_Psi2_pos) ))  
    Error_pos.append(e_pos)

    FT_Psi0_0 = FT_Psi1_0; FT_Psi1_0 = FT_Psi2_0
    FT_Psi0_neg = FT_Psi1_neg; FT_Psi1_neg = FT_Psi2_neg
    FT_Psi0_pos = FT_Psi1_pos; FT_Psi1_pos = FT_Psi2_pos


Iterations = i
x_Iter = np.arange(0, Iterations, 1)

#calculating the spin-components
F_perp = np.sqrt(2) * (np.conj(Psi2_0) * Psi2_pos + np.conj(Psi2_neg) * Psi2_0)
F_z = np.abs(Psi2_pos)**2 - np.abs(Psi2_neg)**2

max_F_perp = np.max(np.abs(F_perp))
max_F_z = np.max(np.abs(F_z))

# showing interesting results
print('Number of Iterations =', Iterations)
print('Number of Resets =', restarts)
print('Number of particles = ', np.sum(rho))

# Plot of the results 
fig, ax = plt.subplots(1, 3, figsize=(15, 5)) 

# First Plot
ax[0].plot(x, np.abs(Psi2_0)**2, color='green', label= f'$\\mu=$ {np.round(mu, 4)} \n' + fr'$\delta =$ {delta}' + f'\n' + f'q = {q}')
ax[0].plot(x, np.abs(Psi2_neg)**2, color='blue', label= r'$\Psi_{-1}$', linestyle = 'dashed')
ax[0].plot(x, np.abs(Psi2_pos)**2, color='red', label= r'$\Psi_{1}$', linestyle = 'dashdot')
ax[0].legend()
ax[0].set_title("Ground state") 
ax[0].set_xlabel("x")
ax[0].set_ylabel(rf"$|\Psi(x)|^2$")

# Second Plot
ax[1].plot(x_Iter, np.abs(Error_0), color='green', label=r"$\Psi_0$")
ax[1].plot(x_Iter, np.abs(Error_neg), color='blue', label=r"$\Psi_{-1}$", linestyle = 'dashed')
ax[1].plot(x_Iter, np.abs(Error_pos), color='red', label=r"$\Psi_{1}$", linestyle = 'dashdot')
ax[1].legend()
ax[1].set_yscale('log')
ax[1].set_title("Residual error") 
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Error")

# Third plot
ax[2].plot(x, np.abs(F_perp), color='black', label=r"$|F_{\perp}|$")
ax[2].plot(x, np.abs(F_z), color='green', label=r"$|F_{z}|$", linestyle = 'dashed')
ax[2].legend()
ax[2].set_title("Spin") 
ax[2].set_ylabel(r"$F_\nu$")
ax[2].set_xlabel("x")
plt.show()