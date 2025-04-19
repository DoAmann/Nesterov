import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn

#define variables and parameters
#define the spatial and Fourier space
L = 750            #length of the spatial space
m = 500*3             #number of descrete points 
dx = 2*L/m          #distance of two points
q = 0.1             #quadratic Zeeman energy
delta = -5e-3       #ratio of c1_1D / c0_1D
Omega = 1e-2        #trep strength
sigma = 500/np.sqrt(2) #width for initial guess

x = np.linspace(-L, L - dx, m)
#k_array = (np.pi / L) * np.concatenate((np.arange(0, m//2), np.arange(-m//2, 0)))  
k_array = 2*np.pi*np.fft.fftfreq(m, dx)

#calculate Laplacian and the potential at the discrete points
def V(x):
    return (1/2) * Omega**2 * x**2
Vx = V(x)
Lap = - k_array**2

#define list for tracing the errors
Error_0 = []
Error_neg = []
Error_pos = []

#initial normalization
def Psi_Initial_0(x):
    return np.exp(-x**2/(2*sigma**2))

def Psi_Initial_neg(x):
    return 0.8*np.exp(-x**2/(2*sigma**2))

def Psi_Initial_pos(x):
    return 0.798*np.exp(-x**2/(2*sigma**2))

#define Psi
N = 20000                           #normalization constant
Psi_i_0 = Psi_Initial_0(x)
Psi_i_neg = Psi_Initial_neg(x)
Psi_i_pos = Psi_Initial_pos(x)

#normalizing
norm_factor = (N/(np.sum(np.abs(Psi_i_0)**2 + np.abs(Psi_i_neg)**2 + np.abs(Psi_i_pos)**2)))**(1/2)

Psi_i_0 = Psi_i_0 * norm_factor            
Psi_i_neg = Psi_i_neg * norm_factor 
Psi_i_pos = Psi_i_pos * norm_factor

#initial iterate
Psi_0 = Psi_i_0;  Psi_neg = Psi_i_neg; Psi_pos = Psi_i_pos
FT_Psi0_0 = fftn(Psi_0); FT_Psi0_neg = fftn(Psi_neg); FT_Psi0_pos = fftn(Psi_pos)
FT_Psi1_0 = FT_Psi0_0; FT_Psi1_neg = FT_Psi0_neg; FT_Psi1_pos = FT_Psi0_pos

#define normalized density
rho = (np.abs(Psi_0)**2 + np.abs(Psi_neg)**2 + np.abs(Psi_pos)**2)

#continue initial iterate
PsiX_0 = Vx * Psi_0 + rho*Psi_0 + delta*(np.abs(Psi_pos)**2 + np.abs(Psi_neg)**2)*Psi_0 + delta*Psi_neg*np.conj(Psi_0)*Psi_pos
PsiX_neg = Vx * Psi_neg + rho*Psi_neg + delta*(np.abs(Psi_neg)**2 + np.abs(Psi_0)**2 - np.abs(Psi_pos)**2)*Psi_neg + delta*Psi_0**2*np.conj(Psi_pos)
PsiX_pos = Vx * Psi_pos + rho*Psi_pos + delta*(np.abs(Psi_pos)**2 + np.abs(Psi_0)**2 - np.abs(Psi_neg)**2)*Psi_pos + delta*Psi_0**2*np.conj(Psi_neg)
FT_PsiX_0 = (1/2)*Lap * FT_Psi1_0 - fftn(PsiX_0)
FT_PsiX_neg = (1/2)*Lap*FT_Psi1_neg - q*FT_Psi1_neg - fftn(PsiX_neg)
FT_PsiX_pos = (1/2)*Lap*FT_Psi1_pos - q*FT_Psi1_pos - fftn(PsiX_pos)

#iteration parameters
dt = 0.5; c = 7               #stepsize and parameter for preconditioner
Restart = 200                     #for the condition of restarting
restarts = 0                    #for counting the number of restarts
ITER = 10000                     #number of maximal iterations
tol=1e-10                   #tolerance
chooser = 2                 #how mu gets calculated
jj = 0; ii = 0; i = 0; e_0=1; e_neg = 1; e_pos = 1

P_inv = (1/(c - Lap))

while np.max(np.real([e_0, e_neg, e_pos])) > tol and (i < ITER):
    i += 1; ii += 1; jj += 1
        
    if chooser == 0:
        #calculate mu
        mu = -np.sum(np.conj(FT_PsiX_0) * P_inv*(FT_Psi1_0)) / np.sum(np.conj(FT_Psi1_0) * P_inv *(FT_Psi1_0)) 
        mu = np.real(mu)
        
    elif chooser == 1:
        #calculate mu with the other equation
        mu = -np.sum(np.conj(FT_PsiX_pos) * P_inv * FT_Psi1_pos) / np.sum(np.conj(FT_Psi1_pos) * P_inv * FT_Psi1_pos)
        mu = np.real(mu)
        
    elif chooser == 2:
        #calculate mu with the other equation
        mu = -np.sum(np.conj(FT_PsiX_neg) * P_inv * FT_Psi1_neg) / np.sum(np.conj(FT_Psi1_neg) * P_inv * FT_Psi1_neg)
        mu = np.real(mu)   

    elif chooser == 3:
        mu_0 = -np.sum(np.conj(FT_PsiX_0) * P_inv*(FT_Psi1_0)) / np.sum(np.conj(FT_Psi1_0) * P_inv *(FT_Psi1_0)) 
        mu_0 = np.real(mu_0)   

        mu_pos = -np.sum(np.conj(FT_PsiX_pos) * P_inv * FT_Psi1_pos) / np.sum(np.conj(FT_Psi1_pos) * P_inv * FT_Psi1_pos)
        mu_pos = np.real(mu_pos) 

        mu_neg = -np.sum(np.conj(FT_PsiX_neg) * P_inv * FT_Psi1_neg) / np.sum(np.conj(FT_Psi1_neg) * P_inv * FT_Psi1_neg)
        mu_neg = np.real(mu_neg)  

        mu = (2*mu_0 + mu_neg + mu_pos)/4
    
    #iteration
    FT_Psi2_0 = (2 - 3/ii) * FT_Psi1_0 + dt**2 * P_inv * (FT_PsiX_0 + mu*FT_Psi1_0) - (1 - 3/ii)*FT_Psi0_0
    Psi2_0 = ifftn(FT_Psi2_0)

    FT_Psi2_neg = (2 - 3/ii) * FT_Psi1_neg + dt**2 * P_inv * (FT_PsiX_neg + mu*FT_Psi1_neg) - (1 - 3/ii)*FT_Psi0_neg
    Psi2_neg = ifftn(FT_Psi2_neg)

    FT_Psi2_pos = (2 - 3/ii) * FT_Psi1_pos + dt**2 * P_inv * (FT_PsiX_pos + mu*FT_Psi1_pos) - (1 - 3/ii)*FT_Psi0_pos
    Psi2_pos = ifftn(FT_Psi2_pos)

    #normalization
    amp = (N/(np.sum(np.abs(Psi2_0)**2 + np.abs(Psi2_neg)**2 + np.abs(Psi2_pos)**2)))**(1/2)  

    Psi2_0 = Psi2_0 * amp
    Psi2_neg = Psi2_neg * amp
    Psi2_pos = Psi2_pos * amp

    FT_Psi2_0 = FT_Psi2_0 * amp
    FT_Psi2_neg = FT_Psi2_neg * amp
    FT_Psi2_pos = FT_Psi2_pos * amp

    #gradient restart
    term1 = np.sum(np.conj(FT_PsiX_pos + mu * FT_Psi1_pos) * (FT_Psi2_pos - FT_Psi1_pos))
    term2 = np.sum(np.conj(FT_PsiX_0 + mu * FT_Psi1_0) * (FT_Psi2_0 - FT_Psi1_0))
    term3 = np.sum(np.conj(FT_PsiX_neg + mu * FT_Psi1_neg) * (FT_Psi2_neg - FT_Psi1_neg))

    cond1 = term1 + term2 + term3
    if cond1 > 0 and ii > Restart:
        ii = 1
        restarts += 1

    #residual error
    rho = (np.abs(Psi2_0)**2 + np.abs(Psi2_neg)**2 + np.abs(Psi2_pos)**2)
    PsiX_0 = Vx * Psi2_0 + rho*Psi2_0 + delta*(np.abs(Psi2_pos)**2 + np.abs(Psi2_neg)**2)*Psi2_0 + delta*Psi2_neg * np.conj(Psi2_0) * Psi2_pos
    PsiX_neg = Vx * Psi2_neg + rho*Psi2_neg + delta*(np.abs(Psi2_neg)**2 + np.abs(Psi2_0)**2 - np.abs(Psi2_pos)**2)*Psi2_neg + delta*Psi2_0**2*np.conj(Psi2_pos)
    PsiX_pos = Vx * Psi2_pos + rho*Psi2_pos + delta*(np.abs(Psi2_pos)**2 + np.abs(Psi2_0)**2 - np.abs(Psi2_neg)**2)*Psi2_pos + delta*Psi2_0**2*np.conj(Psi2_neg)
    FT_PsiX_0 = (1/2)*Lap * FT_Psi2_0 - fftn(PsiX_0) 
    FT_PsiX_neg = (1/2)*Lap*FT_Psi2_neg - q*FT_Psi2_neg - fftn(PsiX_neg)
    FT_PsiX_pos = (1/2)*Lap*FT_Psi2_pos - q*FT_Psi2_pos - fftn(PsiX_pos)
    e_0 = np.sqrt(1/m * np.sum(np.abs(FT_PsiX_0 + mu*FT_Psi2_0)**2 ))  
    e_neg = np.sqrt(1/m * np.sum(np.abs(FT_PsiX_neg + mu*FT_Psi2_neg)**2))  
    e_pos = np.sqrt(1/m * np.sum(np.abs(FT_PsiX_pos + mu*FT_Psi2_pos)**2 ))  
    Error_0.append(e_0)
    Error_neg.append(e_neg)
    Error_pos.append(e_pos)

    
    FT_Psi0_0 = FT_Psi1_0; FT_Psi1_0 = FT_Psi2_0
    FT_Psi0_neg = FT_Psi1_neg; FT_Psi1_neg = FT_Psi2_neg
    FT_Psi0_pos = FT_Psi1_pos; FT_Psi1_pos = FT_Psi2_pos

Iterations = i
x_Iter = np.arange(0, Iterations, 1)

# calculate order parameter
F_perp = np.sqrt(2) * (np.conj(Psi2_0)*Psi2_pos + np.conj(Psi2_neg)*Psi2_0)
F_z = np.abs(Psi2_pos)**2 - np.abs(Psi2_neg)**2

# showing interesting results
print('Number of Iterations =', Iterations)
print('Number of Resets =', restarts)
print('mu = ', np.round(mu, 4), 'delta = ', delta)
print(chooser)
print(np.max(np.abs(F_perp)))

# Plot of the results 
fig, ax = plt.subplots(1, 2, figsize=(12, 5)) 

# First Plot
ax[0].plot(x, np.abs(Psi2_0)**2, color='blue',label = r'$\Psi_0$', linestyle = 'dashdot')
ax[0].plot(x, np.abs(Psi2_neg)**2, color='green',label = r'$\Psi_{-1}$')
ax[0].plot(x, np.abs(Psi2_pos)**2, color='red',label = r'$\Psi_1$', linestyle = 'dotted')
ax[0].plot(x, rho, color = 'grey', label = r'$\rho$')
ax[0].legend()
ax[0].set_title("Ground state") 
ax[0].set_xlabel("x")
ax[0].set_ylabel(r"$|\Psi(x)|^2$")

# Second Plot
ax[1].plot(x_Iter, np.real(Error_0), color='blue', label=r'$\Psi_0$')
ax[1].plot(x_Iter, np.real(Error_neg), color='green', label=r'$\Psi_{-1}$')
ax[1].plot(x_Iter, np.real(Error_pos), color='red', label=r'$\Psi_1$')
ax[1].legend()
ax[1].set_yscale('log')
ax[1].set_title("Residual error") 
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Error")
plt.show()

# Third Plot
plt.plot(x, np.abs(F_z), color = 'green', linestyle = 'dashed', label = r'$F_z$')
plt.plot(x, np.abs(F_perp), color = 'black', label = r'$F_\perp$')
plt.title('Magnetisation')
plt.xlabel('x')
plt.ylabel(r'$|F_\nu|$')
plt.legend()
plt.show()