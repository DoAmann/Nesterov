import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Define variables and parameters
L = 12
m = 128
dx = 2 * L / m
x = np.linspace(-L, L - dx, m)
k_array = (np.pi / L) * np.concatenate((np.arange(0, m//2), np.arange(-m//2, 0)))   
Lap = -k_array**2
def V(x): return 0.1 * x**2
Vx = V(x)

# Initial normalization
def Psi_Int(x): return np.exp(-x**2)
Psi_i = Psi_Int(x)
N = 5
Psi_i *= (N / (np.sum(np.conj(Psi_i) * Psi_i) * dx))**(1/2)

# Define different values to test
sigma_values = [-2, -1, 0, 1, 2]
c = 5.5
results = {}

for sigma in sigma_values:

    # Initialize parameters
    restarts = 0
    i = 0
    ii = 0
    e = 1
    Error = []

     # Initialize wavefunction
    Psi = Psi_i
    FT_Psi0 = fft(Psi)
    FT_Psi1 = FT_Psi0
    FT_PsiX = fft(-Vx * Psi) + fft(sigma * np.abs(Psi)**2 * Psi)

    dt = 0.9
    Restart = 20
    ITER = 1000
    tol = 1e-10

    while e > tol and i < ITER:
        i += 1
        ii += 1

        # Compute mu
        mu = -np.sum(np.conj(Lap * FT_Psi1 + FT_PsiX) * FT_Psi1) / np.sum(np.conj(FT_Psi1) * FT_Psi1) 
        mu = np.real(mu)

        # Iteration update (c is now correctly included!)
        FT_Psi2 = (2 - 3/ii) * FT_Psi1 + dt**2 * (1/(c - Lap)) * (Lap * FT_Psi1 + FT_PsiX + mu * FT_Psi1) - (1 - 3/ii) * FT_Psi0
        Psi2 = ifft(FT_Psi2)

        # Normalization
        amp = (N / (np.sum((np.conj(Psi2) * Psi2)) * dx))**(1/2)  
        Psi2 *= amp
        FT_Psi2 *= amp

        # Residual error
        FT_PsiX = fft(-Vx * Psi2 + sigma * (np.abs(Psi2)**2) * Psi2)
        e = np.sqrt(dx/m * np.sum((FT_PsiX + mu * FT_Psi2 + Lap * FT_Psi2) * np.conj(FT_PsiX + mu * FT_Psi2 + Lap * FT_Psi2)))  
        Error.append(e)

        # Restart condition
        cond1 = np.sum((Lap * FT_Psi1 + FT_PsiX + mu * FT_Psi1) * np.conj(FT_PsiX - FT_Psi1))
        if cond1 > 0 and ii > Restart:
            ii = 1
            restarts += 1
            
        FT_Psi0 = FT_Psi1
        FT_Psi1 = FT_Psi2

        results[sigma] = {"Psi2": np.real(Psi2), "mu": mu, "Error": np.real(Error), "Iterations": i, "Restarts": restarts}

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(r"Calculation for $V(x) = 0.1x^2$", fontsize=14)


# Plot wavefunctions for different c and sigma
for sigma in results:
    ax[0].plot(x, results[sigma]["Psi2"], label=f'σ = {sigma}, μ = {np.round(results[sigma]["mu"], 4)}')

ax[0].set_title("Ground States for Different σ")
ax[0].set_xlabel("x")
ax[0].set_ylabel(r"$\Psi(x)$")
ax[0].legend()

# Plot residual errors
for sigma in results:
    ax[1].plot(np.arange(results[sigma]["Iterations"]), results[sigma]["Error"], label=f'σ = {sigma}')

ax[1].set_yscale('log')
ax[1].set_title("Error Decay for Different σ")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Error")
ax[1].legend()

plt.show()

#Again for a different potential
# Define variables and parameters
L = 12
m = 128
dx = 2 * L / m
x = np.linspace(-L, L - dx, m)
k_array = (np.pi / L) * np.concatenate((np.arange(0, m//2), np.arange(-m//2, 0)))   
Lap = -k_array**2
def V(x): return 0.01 * x**4 - 0.02 * x**2
Vx = V(x)

# Initial normalization
def Psi_Int(x): return np.exp(-x**2)
Psi_i = Psi_Int(x)
N = 5
Psi_i *= (N / (np.sum(np.conj(Psi_i) * Psi_i) * dx))**(1/2)

# Define different values to test
sigma_values = [-2, -1, 0, 1, 2]
c = 5.5
results = {}

for sigma in sigma_values:

    # Initialize parameters
    restarts = 0
    i = 0
    ii = 0
    e = 1
    Error = []

     # Initialize wavefunction
    Psi = Psi_i
    FT_Psi0 = fft(Psi)
    FT_Psi1 = FT_Psi0
    FT_PsiX = fft(-Vx * Psi) + fft(sigma * np.abs(Psi)**2 * Psi)

    dt = 0.3
    Restart = 24
    ITER = 1000
    tol = 1e-10

    while e > tol and i < ITER:
        i += 1
        ii += 1

        # Compute mu
        mu = -np.sum(np.conj(Lap * FT_Psi1 + FT_PsiX) * FT_Psi1) / np.sum(np.conj(FT_Psi1) * FT_Psi1) 
        mu = np.real(mu)

        # Iteration update (c is now correctly included!)
        FT_Psi2 = (2 - 3/ii) * FT_Psi1 + dt**2 * (1/(c - Lap)) * (Lap * FT_Psi1 + FT_PsiX + mu * FT_Psi1) - (1 - 3/ii) * FT_Psi0
        Psi2 = ifft(FT_Psi2)

        # Normalization
        amp = (N / (np.sum((np.conj(Psi2) * Psi2)) * dx))**(1/2)  
        Psi2 *= amp
        FT_Psi2 *= amp

        # Residual error
        FT_PsiX = fft(-Vx * Psi2 + sigma * (np.abs(Psi2)**2) * Psi2)
        e = np.sqrt(dx/m * np.sum((FT_PsiX + mu * FT_Psi2 + Lap * FT_Psi2) * np.conj(FT_PsiX + mu * FT_Psi2 + Lap * FT_Psi2)))  
        Error.append(e)

        # Restart condition
        cond1 = np.sum((Lap * FT_Psi1 + FT_PsiX + mu * FT_Psi1) * np.conj(FT_PsiX - FT_Psi1))
        if cond1 > 0 and ii > Restart:
            ii = 1
            restarts += 1
            
        FT_Psi0 = FT_Psi1
        FT_Psi1 = FT_Psi2

        results[sigma] = {"Psi2": np.real(Psi2), "mu": mu, "Error": np.real(Error), "Iterations": i, "Restarts": restarts}

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(r"Calculation for $V(x) = 0.01x^4 - 0.02x^2$", fontsize=14)


# Plot wavefunctions for different sigma
for sigma in results:
    ax[0].plot(x, results[sigma]["Psi2"], label=f'σ = {sigma}, μ = {np.round(results[sigma]["mu"], 4)}')

ax[0].set_title("Ground States for Different σ")
ax[0].set_xlabel("x")
ax[0].set_ylabel(r"$\Psi(x)$")
ax[0].legend()

# Plot residual errors
for sigma in results:
    ax[1].plot(np.arange(results[sigma]["Iterations"]), results[sigma]["Error"], label=f'σ = {sigma}')

ax[1].set_yscale('log')
ax[1].set_title("Error Decay for Different σ")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Error")
ax[1].legend()

plt.show()
