#region notes
'''
'''
#endregion

#region import libraries 

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.special as special
import math
import os
import glob
import h5py as h
import re
import time
import json
import cupy as cp
from pathlib import Path
from cupy.fft import fft, ifft
import time
import shutil
import signal
import sys
sys.path.insert (1, "../Libraries")
import utils
import Correlators
sys.path.insert (1, "1D_Libraries")
from Propagators1D import *
import WaveFuncGen1D
sys.path.insert (1, '../')

from cupyx.scipy.fft import get_fft_plan
from cupyx.scipy.fft import fft, ifft
#endregion

#region Parameter Definitions

datapath = "Bidirectional_Scaling" # Folder name for simulation (Run# appended at the end)

activate_noise = 1 # Truncated Wigner noise
activate_trap = 0 # 0- homogenous gas, 1- harmonic trap
trap_geo = 'boxtrap' # trap geometry. accepted geometries: harmonic and boxtrap
only_eval_trap = 0 # 0- evaluate entire grid, 1 evaluate between N_points/2-evalsize/2 and N_points/2 + evalsize/2
center_around_0 = 1 
number_of_runs = 1 # how many runs to be calculated at parallel (per file)
number_of_different_files = 2 # number of different parallely computed runs (serial)
save_grid = 1 # save the fundamental field data in h5 file
data_analysis = 1 #compute averages and save
phase = "P" #quantum phase. Accepted phases: P, EP, AF (all particles at mF=1), F

if (not activate_trap):
    only_eval_trap = 0

N_points = 4096 # number of numerical grid points
evalsize = 480 # number of gridpoints to evaluate in data analysis (centered around middle of grid)

m = 1.44*10**(-25) # atom mass
l = 220e-6 # physical length in meters
N_par = 3e6 # number of particles


omega_parallel = 2* cp.pi * 2.5 # Trapping frequency in longitudinal direction in Hz
w_perp =  2 * cp.pi * 250 #  Trapping frequency in transverse direction in Hz (isotropic in transverse direction)

t_scaling = "healing_time" # units for time. "healing_time" for spin healing_time and "seconds" for SI units
num_snaps = 200 # number of time snapshots to be saved
dt = 0.2 # timestep in numerical units. Please always choose dt<1
t_offset = 0 # offset for snapshot saving
t_end = 200# total integration time, in units of t_scaling


eq_diff = 1e-12 # convergence zone for imaginary time propagation for ground state
number_of_hermite_polynoms = 100 # how many hermite polynomials to include in TW for a trap

q_scaling = 'nc1' # units for q. "nc1" or "Hz". nc1 will be q/nc1. Hz means it will be 2pi*hbar*q
q = 0.9 # quadratic Zeeman shift for propagation in units of n|c_1|
q_init = 0.9 # initial Zeeman shift for initial condition (trap and in the EP)

number_of_steps_imaginary_time_propagation = 200

## Calculate other basic properties
a_B = const.physical_constants ["Bohr radius"] [0] 
a0 = 101.8*a_B #Scattering length 0 channel
a2 = 100.4*a_B #Scattering length 2 channel
n = N_par / N_points # Density
dx = l / N_points # Length scale
x_vals = cp.arange(0, N_points) # grid
x_prime = x_vals - (N_points / 2) * center_around_0 # center grid around 0 for trap
k = 2 * cp.sin(cp.pi * cp.arange(0, N_points) / (N_points)) # momentum grid

## Calculate interaction coefficients
a_HO = cp.sqrt(const.hbar/(m*w_perp)) # harmonic oscillator length of transverse trap
c0 = 4*cp.pi*const.hbar**2/(3*m)*(a0+2*a2)/(2*cp.pi*a_HO**2) # density-density interactions
c0 *= m *dx/ (const.hbar**2) # make unitless
c1 = -0.01* c0 # already unitless, spin-spin interactions

healing_time = 2*cp.pi/(n*cp.abs(c1))# spin interaction time
Tscale = m * dx**2/const.hbar # Time scale 

## Scale time according to healing time or seconds
if (t_scaling == "healing_time"):
    t_offset *= healing_time
    t_end *= healing_time

elif(t_scaling == "seconds"):
    t_offset /= Tscale
    t_end /= Tscale



## Trap 
omega_parallel *= Tscale # make frequency unitless
if activate_trap:
    if (trap_geo=='harmonic'):
        V = (1/2 * (omega_parallel * x_prime)**2).astype(cp.float64) # Harmonic potential
    elif(trap_geo =='boxtrap'):
        V = 1.e10/(1.+cp.exp(-8.*(x_prime-evalsize//2)/15)) + 1.e10/(1.+cp.exp(8.*(x_prime+evalsize//2)/15))
else:
    V = cp.zeros (N_points)
    omega_parallel=0

# make q unitless
if(q_scaling=='nc1'):
    q *= (-n*c1)
    q_init *= (-n*c1)

if (q_scaling == "Hz"):
    q *= 2*np.pi*Tscale
    q_init *= 2*np.pi*Tscale


# Transfer parameters to device
c0 = cp.float64(c0)
c1 = cp.float64(c1)
dt = cp.float64(dt)
q = cp.float64(q)
t_end = cp.float64(t_end)
t_offset = cp.float64(t_offset)
#endregion


time_start_allruns = time.time () # start timer


if (only_eval_trap):
    corr_fp_k = cp.zeros((num_snaps, evalsize)) # initialize container for transverse spin spectra
else:
    corr_fp_k = cp.zeros((num_snaps, N_points)) # initialize container for transverse spin spectra

t_snap = np.linspace(t_offset / dt, t_end /
                    dt, num_snaps, endpoint=True, dtype=int) #snapshot array

## Initialize k propagation with fft normalization already
prop_phase = k ** 2 * dt / 2.
K_prop = cp.exp (-1j * prop_phase)
K_prop_imag_time = cp.exp (-prop_phase)

## Set gpu settings
grid_size, block_size, block_size_imag, grid_size_imag = utils.gpu_setup1D (N_points, number_of_runs)

new_dir = utils.init_data_dir(datapath) # initiate directory 

#region Real time propagation and save

def cleanup():
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
        print(f"Deleted folder: {new_dir}")

def handle_sigterm(signum, frame):
    print("Job terminated by SLURM (SIGTERM). Cleaning up...")
    cleanup()
    sys.exit(0)

# Register the SIGTERM handler
signal.signal(signal.SIGTERM, handle_sigterm)

try:
    for file_num in range (number_of_different_files):

        print (f"Generating file {file_num + 1}/{number_of_different_files}")

        psi = WaveFuncGen1D.init_psi(phase, number_of_runs, N_points, activate_noise, N_par, n, q_init, c1, \
                eq_diff, K_prop_imag_time, V, block_size_imag, grid_size_imag, activate_trap, trap_geo, evalsize,\
                number_of_steps_imaginary_time_propagation, c0, dt, omega_parallel, x_prime, number_of_hermite_polynoms) # initialize fields        
        

        psi_arr = cp.zeros((num_snaps, 3, number_of_runs, N_points), dtype= cp.complex128) # container for raw data
        time_start_run = time.time () # start run timer
        
        count = 0 # snapshot count
        t= 0 # time count
        
        while (t < (t_end/ dt) - 1):
            
            # iterate over snapshots
            while(t < (t_snap[count])):
                
                psi = fft (psi, axis= 2) # transform to momentum space
                
                kpropagate(grid_size, block_size, 
                (number_of_runs,N_points,psi[0], psi[1],psi[2], K_prop)) # momentum space propagation

                psi = ifft(psi, axis= 2) # transform back to position space (can we do the normalization in a smart way?)

                rpropagate_diag(grid_size, block_size ,
                (number_of_runs,N_points,psi[0], psi[1],psi[2], c1, c0,q, dt, V)) # real space propagation diagonal part half step

                rpropagate_non_diag(grid_size, block_size ,
                (number_of_runs,N_points,psi[0], psi[1],psi[2], c1, dt)) # real space propagation non-diagonal part

                rpropagate_diag(grid_size, block_size ,
                (number_of_runs,N_points,psi[0], psi[1],psi[2], c1, c0,q, dt, V)) # real space propagation diagonal part half step
                
                t = t + 1

            psi_arr[count] = cp.copy(psi) # copy to large array

            count = count + 1
            print("propagating " + str(count) + "/" + str (num_snaps) + " done")
        
        

        h5attributes = {"l": l, "N_points": N_points, "N_par": N_par,\
        "number_of_hermite_polynoms": number_of_hermite_polynoms, \
        "dx": dx, "Tscale": Tscale, "healing_time": float (healing_time),\
        "runs_per_file": number_of_runs, "dt": float (dt), 'q_init': float (q_init), 
        "q": float (q), "rho": float(n), "num_snaps": float (num_snaps),\
        "eq_diff": float (eq_diff), 'omega_par': float (omega_parallel), 'omega_perp': float (w_perp),  \
        'activate_trap': float (activate_trap), \
        'activate_noise': float (activate_noise), 'phase': phase, 't_offset': float(t_offset) \
        , 't_end': float(t_end), 'num_steps_imag_time_prop': float (number_of_steps_imaginary_time_propagation), \
        'c0': float (c0), 'c1': float (c1), 'only_eval_trap': int(only_eval_trap), 'evalsize': float(evalsize)
        } # parameters for savings

        file_name = str(file_num) # append number to filename

        
        if(data_analysis):
            Fpsq_k = Correlators.F_p(psi_arr, activate_trap, N_points, only_eval_trap, evalsize) # calculate transverse spin spectrum
            corr_fp_k += Fpsq_k / number_of_different_files # average over the different files
        
        
        if(save_grid):
            
            utils.init_h5_file (new_dir, file_name, psi_arr, h5attributes) # save the raw grid data
            psi_arr = cp.swapaxes (psi_arr, 0, 2) # swap axis for array to be (runs, spin, time, space)

            print (f"File {file_num + 1}/{number_of_different_files} generated succesfully")
        del psi_arr
        time_end_run = time.time ()
        time_run = time_end_run - time_start_run
        print ("Run time = " + str (np.round (time_run, 2)) + " seconds")
        print ("")

    if(data_analysis):
        utils.save_means(new_dir,h5attributes, corr_fp_k) #save the averaged spectra

    time_end_allruns = time.time ()
    allruns_time = time_end_allruns - time_start_allruns 

    print ("Total run time = " + str (np.round (allruns_time, 2)) + " seconds")
    print (f"Directory: {new_dir}")

## Give exceptions to delete data in case of premature exit of the code
except KeyboardInterrupt:
    print("Loop interrupted by user.")
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
        print(f"Deleted folder: {new_dir}")
except Exception as e:
    print(f"An error occurred: {e}")
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
        print(f"Deleted folder: {new_dir}")
#endregion