import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn 
import sympy as sp #just for the plot

#define variables and parameters
#define the potential
x = sp.Symbol('x')
def V(x):
    return 0.1*x**2

V_str = sp.latex(V(x))

#define the spatial and Fourier space
L = 12              #length of the spatial space
m = 128             #number of descrete points 
dx = 2*L/m          #distance of two points
sig = 1            #parameter in the equation -> can be changed 

x = np.linspace(-L, L - dx, m)
k_array = (np.pi / L)  * np.concatenate((np.arange(0, m//2), np.arange(-m//2, 0)))   

#calculate Laplacian and the potential at the discrete points
Lap = -k_array**2
Vx = V(x)

#define list for tracing the errors
Error = []

#initial normalization
def Psi_Initial(x):
    return np.exp(-x**2)

#normalizing
N = 5                           #normalization constant
Psi_i = Psi_Initial(x)
Psi_i = Psi_i * (N/(dx * np.sum(np.conj(Psi_i) * Psi_i)))**(1/2)  

#initial iterate
Psi = Psi_i
FT_Psi0 = fftn(Psi)
FT_Psi1 = FT_Psi0
FT_PsiX = fftn(-Vx * Psi + sig*np.abs(Psi)**2*Psi) 

#iteration parameters
dt = 0.9; c = 4;               #stepsize and parameter for preconditioner
Restart = 23                     #for the condition of restarting
restarts = 0                    #for counting the number of restarts
ITER = 1000                     #number of maximal iterations
tol=10**(-10)                   #tolerance
jj = 0; ii = 0; i = 0; e=1

while e>tol and i < ITER:
    i += 1; ii += 1; jj += 1

    #calculate mu
    mu = -np.sum(np.conj(Lap * FT_Psi1 + FT_PsiX) * ((FT_Psi1))) / np.sum(np.conj(FT_Psi1) * (FT_Psi1)) 
    mu = np.real(mu)

    #iteration
    FT_Psi2 = (2 - 3/ii) * FT_Psi1 + dt**2 * (1/(c - Lap)) * (Lap * FT_Psi1 + FT_PsiX + mu*FT_Psi1) - (1 - 3/ii)*FT_Psi0
    Psi2 = ifftn(FT_Psi2)

    #normalization
    amp = (N/(np.sum((np.conj(Psi2) * Psi2))*dx))**(1/2)  
    Psi2 = Psi2 * amp
    FT_Psi2 = FT_Psi2 * amp

    #gradient restart
    cond1 = np.sum((Lap * FT_Psi1 + FT_PsiX + mu * FT_Psi1) * np.conj(FT_PsiX - FT_Psi1))
    if cond1 > 0 and ii > Restart:
        ii = 1
        restarts += 1

    #residual error
    FT_PsiX = fftn(-Vx * Psi2 + sig * (np.abs(Psi2)**2) * Psi2)
    e = np.sqrt(dx/m * np.sum((FT_PsiX + mu*FT_Psi2 + Lap * FT_Psi2) * np.conj(FT_PsiX + mu * FT_Psi2 + Lap * FT_Psi2) ))  
    Error.append(e)

    FT_Psi0 = FT_Psi1; FT_Psi1 = FT_Psi2


Iterations = i
x_Iter = np.arange(0, Iterations, 1)

# showing interesting results
print('Number of Iterations =', Iterations)
print('Number of Resets =', restarts)

# Plot of the results 
fig, ax = plt.subplots(1, 2, figsize=(12, 5)) 

# First Plot
ax[0].plot(x, np.real(Psi2), color='blue', label= rf'$V(x) = {V_str}$' + f'\n$\\mu=$ {np.round(mu, 4)} \n' + fr'$\sigma =$ {sig}')
ax[0].legend()
ax[0].set_title("Ground state") 
ax[0].set_xlabel("x")
ax[0].set_ylabel(rf"$\Psi(x)$")

# Second Plot
ax[1].plot(x_Iter, np.real(Error), color='red', label="ACTN")
ax[1].legend()
ax[1].set_yscale('log')
ax[1].set_title("Residual error") 
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Error")
plt.show()

#is it okay to take only the real part?