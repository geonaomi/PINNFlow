## Physics-informed NN core flow inversion ##
## Author: Naomi Shakespeare-Rees ## 
## Last updated: 29 October 2024 ##

import numpy as np
import chaosmagpy as cp
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time 

def data_generator(file_name, year, month =1, day =1, Nmax = 13, dy = 30, dx = 55, clat1 = None, clat2 = None, long1 = None, long2 = None):
	r"""
	Computes Radial Magnetic Field, Secular Variation, 
	and the theta, phi components of the horizontal 
	divergence of the Radial Magnetic Field, evaluated on a grid 
	box using Gauss Coefficients from the CHAOS model. 
	Each value is then rescaled so that all components in loss function 
	are of order one. For more information on this, please see Notes.
	
	Parameters
	----------
	file_name : str
		Filepath and name of the MAT-file
		
	year : int, ndarray
	month : int, ndarray, optional
		Defaults to 1 (January)
	day : int, ndarray, optional
		Defaults to 1
	Nmax : int, positive
		Maximum Degree of the Spherical Harmonic Expansion, default 13
	dy : int, positive, optional
		Number of grid points in the theta direction, default 30
	dx : int, positive, optional
		Number of grid points in the phi direction, default 55
	clat1 : ndarray, float
		Colatitude, in degrees, of the upper boundary of the grid box
	clat2 : ndarray, float
		Colatitude, in degrees, of the lower boundary of the grid box	
	long1 : ndarray, float
		Longitude, in degrees, of the left boundary of the grid box
	long2 : ndarray, float
		Longitude, in degrees, of the right boundary of the grid box
		
	Returns
	-------
	radius_data, theta_data, phi_data : ndarray, shape (...)
		Radial, Theta, and Phi coordinates, each of shape (dx, dy).
		Theta and Phi coordinates given in radians. 
	Br_data : ndarray, shape (...)
		Radial field component, given in $\mu T$
	Br_data_dot : ndarray, shape (...)
		Radial Secular Variation, given in $\mu T/0.1 year$
	horiz_div_theta : ndarray, shape (...)
		Theta component of the horizontal divergence of the radial field component,
		given in $\mu T/km$
	horiz_div_phi
		Phi component of the horizontal divergence of the radial field component,
		given in $\mu T/km$
		
	Notes
	-----
	Rescaling needed to ensure all inputs and outputs to the PINN are of the order one.
	To do this:
		- Br_data is rescaled from $nT$ to $\mu T$, which puts it at order ~$10^3$
		- Time is rescaled from $year$ to $0.1 year$
		- Br_dot_data is rescaled from $nT/year$ to $\mu T/0.1 year$
		- horiz_div_theta, horiz_div_phi are recaled from $nT/m$ to $\mu T/km$
	
	"""
	
	radius = 3485
	cl1 = int(clat1)
	cl2 = int(clat2)
	l1 = int(long1)
	l2 = int(long2)
	theta = np.linspace(cl1, cl2, num=dy) # colatitude in degrees
	phi = np.linspace(l1, l2, num=dx) # longitude in degrees
	phi_grid, theta_grid = np.meshgrid(phi, theta)
	radius_grid = radius*np.ones(phi_grid.shape) #grid of shape (dx, dy), with all values equal to radius
	phi_data,theta_data = np.meshgrid(np.radians(phi),np.radians(theta)) #colat, longitude in radians
	
	#Computes the modified Julian date as floating point number
	time = cp.data_utils.mjd2000(year, month, day)
	#Loading in model, with spherical harmonic degree Nmax
	model = cp.load_CHAOS_matfile(file_name)
	# Values on grid
	Br, _,_ = model.synth_values_tdep(time, radius, theta, phi, grid=True,deriv = 0,nmax=Nmax) #in nT
	Br_dot, _,_  = model.synth_values_tdep(time, radius, theta, phi, grid=True,deriv=1, nmax=Nmax) #in nT/year
	#Compute Gauss coefficients for spatial derivatives 
	gauss = model.synth_coeffs_tdep(time, nmax=Nmax, deriv = 0) 
	gauss_ = gauss.copy()
	k = 0
	for l in range(1,Nmax+1): #l=1,2,3,...Nmax
		for j in range(2*l+1): #l=1 has 3 coeffs, l=2 has 5 coefficients...
			gauss_[k] = - gauss_[k] * (l+1) /(radius*10**3)
			k+= 1
	#Spatial derivatives on grid
	horiz_div_rad, horiz_div_theta, horiz_div_phi = cp.model_utils.synth_values(gauss_, radius, theta, phi, grid=True) #in nT/m

	Br_data = Br/1e3 #units in microT
	Br_dot_data = Br_dot/1e4 #units in microT/0.1year

	return(radius_grid, phi_data, theta_data, Br_data, Br_dot_data, horiz_div_theta, horiz_div_phi)