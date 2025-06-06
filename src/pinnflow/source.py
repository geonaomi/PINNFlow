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
import h5py
import warnings

def data_generator(file_name, year, month =1, day =1, Nmax = 13, dy = 30, dx = 55, clat1 = None, clat2 = None, long1 = None, long2 = None, mat_file = None):
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
	if file_name.endswith(".hdf5"):
		if mat_file == None:
			raise FileNotFoundError("No Matfile supplied, please supply matfile in format of CHAOS to proceed.")
		else:
			f = h5py.File(file_name, 'r')['SOLA']
			snapshot = f['SNAPSHOT_'+str(year)+'_'+str(month)]
			Br_dot = np.array(snapshot['sv'][1: -1])
			theta_data = np.array(snapshot['theta'][1: -1])*np.pi/180
			phi_data = (np.array(snapshot['phi'][1: -1]))*np.pi/180
			radius_grid = radius*np.ones(phi_data.shape)
			theta_deg = np.array(snapshot['theta'][1: -1])
			phi_deg = np.array(snapshot['phi'][1: -1])
			time = cp.data_utils.mjd2000(year, month, day)
			model = cp.load_CHAOS_matfile(mat_file)
			# Values on grid
			Br, _,_ = model.synth_values_tdep(time, radius, theta_deg, phi_deg, grid=False,deriv = 0,nmax=Nmax)
			gauss = model.synth_coeffs_tdep(time, nmax=Nmax, deriv = 0) 
			gauss_ = gauss.copy()
			k = 0
			for l in range(1,Nmax+1): #l=1,2,3,...Nmax
				for j in range(2*l+1): #l=1 has 3 coeffs, l=2 has 5 coefficients...
					gauss_[k] = - gauss_[k] * (l+1) /(radius*10**3)
					k+= 1
			#Spatial derivatives on grid
			horiz_div_rad, horiz_div_theta, horiz_div_phi = cp.model_utils.synth_values(gauss_, radius, theta_deg, phi_deg, grid=False) #in nT/m
	
	else:
		cl1 = int(clat1)
		cl2 = int(clat2)
		l1 = int(long1)
		l2 = int(long2)
		theta_deg = np.linspace(cl1, cl2, num=dy, endpoint=True) # colatitude in degrees
		phi_deg = np.linspace(l1, l2, num=dx, endpoint=False) # longitude in degrees
		phi_grid, theta_grid = np.meshgrid(phi_deg, theta_deg)
		radius_grid = radius*np.ones(phi_grid.shape) #grid of shape (dx, dy), with all values equal to radius
		phi_data,theta_data = np.meshgrid(np.radians(phi_deg),np.radians(theta_deg)) #colat, longitude in radians
		
		#Computes the modified Julian date as floating point number
		time = cp.data_utils.mjd2000(year, month, day)
		#Loading in model, with spherical harmonic degree Nmax
		model = cp.load_CHAOS_matfile(file_name)
		# Values on grid
		Br, _,_ = model.synth_values_tdep(time, radius, theta_deg, phi_deg, grid=True,deriv = 0,nmax=Nmax) #in nT
		Br_dot, _,_  = model.synth_values_tdep(time, radius, theta_deg, phi_deg, grid=True,deriv=1, nmax=Nmax) #in nT/year
		#Compute Gauss coefficients for spatial derivatives 
		gauss = model.synth_coeffs_tdep(time, nmax=Nmax, deriv = 0) 
		gauss_ = gauss.copy()
		k = 0
		for l in range(1,Nmax+1): #l=1,2,3,...Nmax
			for j in range(2*l+1): #l=1 has 3 coeffs, l=2 has 5 coefficients...
				gauss_[k] = - gauss_[k] * (l+1) /(radius*10**3)
				k+= 1
		#Spatial derivatives on grid
		horiz_div_rad, horiz_div_theta, horiz_div_phi = cp.model_utils.synth_values(gauss_, radius, theta_deg, phi_deg, grid=True) #in nT/m
		
	Br_data = Br/1e3 #units in microT
	Br_dot_data = Br_dot/1e4 #units in microT/0.1year
	
	if any(np.round(theta) == 90.0 for theta in theta_deg):
		
		warnings.warn("Warning, theta_data contains values at the equator. This is not compatible with the TG flow assumption.", stacklevel=3)
		inp = input(" Theta_data contains values at the equator. This is not compatible with the TG flow assumption, and will need to be removed  \n in order for the TG flow assumption to be applied. \n \n Would you like these values to be removed? This is not needed if you are not applying TG.  \n \n   'yes' or 'no'")
		if inp == 'yes':
			if Br_dot[0,:].shape[0] == 1:
				radius_grid = np.delete(radius_grid, np.where(np.round(theta_deg)==90.0))
				theta_data = np.delete(theta_data, np.where(np.round(theta_deg)==90.0))
				phi_data = np.delete(phi_data, np.where(np.round(theta_deg)==90.0))
				Br_data = np.delete(Br_data, np.where(np.round(theta_deg)==90.0))
				Br_dot_data = np.delete(Br_dot_data, np.where(np.round(theta_deg)==90.0))
				horiz_div_theta =  np.delete(horiz_div_theta, np.where(np.round(theta_deg)==90.0))
				horiz_div_phi =  np.delete(horiz_div_phi, np.where(np.round(theta_deg)==90.0))
			else:
				theta_ = np.delete(theta_deg, np.where(np.round(theta_deg)==90.0))
				phi_data,theta_data = np.meshgrid(np.radians(phi_deg),np.radians(theta_))
				phi_test,theta_test = np.meshgrid(np.radians(phi_deg),np.radians(theta_deg))
				coord = theta_test.flatten()
				radius_grid = np.delete(radius_grid.flatten(), np.where(np.round(coord*180/np.pi)==90.0)).reshape(phi_data.shape)
				Br_data = np.delete(Br_data.flatten(), np.where(np.round(coord*180/np.pi)==90.0)).reshape(phi_data.shape)
				Br_dot_data = np.delete(Br_dot_data.flatten(), np.where(np.round(coord*180/np.pi)==90.0)).reshape(phi_data.shape)
				horiz_div_theta =  np.delete(horiz_div_theta.flatten(), np.where(np.round(coord*180/np.pi)==90.0)).reshape(phi_data.shape)
				horiz_div_phi =  np.delete(horiz_div_phi.flatten(), np.where(np.round(coord*180/np.pi)==90.0)).reshape(phi_data.shape)
		else:
			pass
			
	

	return(radius_grid, phi_data, theta_data, Br_data, Br_dot_data, horiz_div_theta, horiz_div_phi)
	
class PINNFlow(nn.Module):
    r"""
    Computes Toroidal and Poloidal Scalars, which are then differentiated to find flows.
    Solutions should match both the input SV (sv_out = SV_CHAOS) and the 
    flows constraint (tg_con = 0).
    
    Parameters
    ----------
    radius_input : torch.Tensor, ndarray
        Radial coordinates at the Core-Mantle Boundary, of shape (flattened(dx,dy), 1). 
        Units of km.
    theta_input : torch.Tensor, ndarray
        Latitude coordinates, of shape (flattened(dx,dy), 1). 
        Units of Radians.
    phi_input : torch.Tensor, ndarray
        Longitude coordinates, of shape (flattened(dx,dy), 1). 
        Units of Radians.
        
	horiz_div_theta : torch.Tensor, ndarray
		Theta component of the horizontal divergence of the radial field component.
		Units of $\mu T/km$.
	horiz_div_phi : torch.Tensor, ndarray
		Phi component of the horizontal divergence of the radial field component.
		Units of $\mu T/km$.
    br_input : torch.Tensor, ndarray
		Radial magnetic field component.
        Units of $\mu T$.
    model : __main__.NeuralNet
        Initialised Pytorch Network, used to compute the Toroidal and Poloidal Scalars.
        
    Returns
    -------
    u_theta : torch.Tensor
        torch.Tensor representing the theta component of the flow.
        To evaluate on a grid, use u_theta.flatten().detach().numpy() to convert to an 
        ndarray of shape  (flattened(dx,dy), 1), and re-shape to original grid shape using np.reshape(...). 
    u_phi : torch.Tensor
        torch.Tensor representing the phi component of the flow.
        To evaluate on a grid, use u_phi.flatten().detach().numpy() to convert to an 
        ndarray of shape  (flattened(dx,dy), 1), and re-shape to original grid shape using np.reshape(...). 
    div_uh : torch.Tensor
        torch.Tensor representing the horizontal divergence of the flow. 
        To evaluate on a grid, use div_uh.flatten().detach().numpy() to convert to an 
        ndarray of shape  (flattened(dx,dy), 1), and re-shape to original grid shape using np.reshape(...). 
    sv_out : torch.Tensor
        torch.Tensor representing the predicted SV from the calculated flows. 
        To evaluate on a grid, use sv_out.flatten().detach().numpy() to convert to an 
        ndarray of shape  (flattened(dx,dy), 1), and re-shape to original grid shape using np.reshape(...). 
    tg_con : torch.Tensor
        torch.Tensor representing the Tangentially Geostophic flow condition. 
        To evaluate on a grid, use tg_con.flatten().detach().numpy() to convert to an 
        ndarray of shape  (flattened(dx,dy), 1), and re-shape to original grid shape using np.reshape(...). 
    comp : torch.Tensor
        torch.Tensor representing the complexity, as defined by Bloxham 1988. 
        To evaluate on a grid, use comp.flatten().detach().numpy() to convert to an 
        ndarray of shape  (flattened(dx,dy), 1), and re-shape to original grid shape using np.reshape(...). 
        
    Notes
    -----
    """
    # Initialising Class
    def __init__(self, radius_input, phi_input, theta_input, horiz_theta_input, horiz_phi_input, br_input, model):
        super().__init__()
        if isinstance(radius_input, torch.Tensor):
            self.radius = radius_input
            self.phi = phi_input
            self.theta = theta_input
            self.horiz_theta = horiz_theta_input
            self.horiz_phi = horiz_phi_input
            self.br = br_input
        # If inputs are not torch.Tensors, convert them.
        else:
            self.radius = torch.tensor(radius_input, requires_grad=True).float()
            self.phi = torch.tensor(phi_input, requires_grad=True).float()
            self.theta = torch.tensor(theta_input, requires_grad=True).float()
            self.horiz_theta = torch.tensor(horiz_theta_input, requires_grad=True).float()
            self.horiz_phi = torch.tensor(horiz_phi_input, requires_grad=True).float()
            self.br = torch.tensor(br_input, requires_grad=True).float()
        self.model = model
    def net_sv(self):
        
        self.theta.requires_grad_(True)
        self.phi.requires_grad_(True)
        #Calculating Toroidal and Poloidal Scalars from input NN models 
        X = torch.cat([self.phi, self.theta], dim=1)
        T =  self.model(X)
        P =  self.model(X)

        # Differentiating T, P
        T_theta = torch.autograd.grad(T, self.theta, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True )[0]
        P_theta = torch.autograd.grad(P, self.theta, grad_outputs=torch.ones_like(P), create_graph=True, retain_graph=True )[0]

        T_phi = torch.autograd.grad(T, self.phi, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True )[0]
        P_phi = torch.autograd.grad(P, self.phi, grad_outputs=torch.ones_like(P), create_graph=True, retain_graph=True )[0]
    
        #Horizontal Flow at the CMB
        u_theta = (T_phi/torch.sin(self.theta)) + P_theta
        u_phi = (P_phi/torch.sin(self.theta))-T_theta 
        
        #Calculating Horizontal Divergence
        u_theta_theta = torch.autograd.grad(-u_theta*torch.sin(self.theta), self.theta, grad_outputs=torch.ones_like(u_theta), create_graph=True, retain_graph=True )[0]
        u_theta_theta_theta = torch.autograd.grad(u_theta_theta*torch.sin(self.theta), self.theta, grad_outputs=torch.ones_like(u_theta), create_graph=True, retain_graph=True )[0]
        
        u_theta_phi = torch.autograd.grad(-u_theta, self.phi, grad_outputs=torch.ones_like(u_theta), create_graph=True, retain_graph=True )[0]
        u_theta_phi_phi =  torch.autograd.grad(u_theta_phi, self.phi, grad_outputs=torch.ones_like(u_theta), create_graph=True, retain_graph=True )[0]
        
        u_phi_theta = torch.autograd.grad(u_phi*torch.sin(self.theta), self.theta, grad_outputs=torch.ones_like(u_theta), create_graph=True, retain_graph=True )[0]
        u_phi_theta_theta = torch.autograd.grad(u_phi_theta*torch.sin(self.theta), self.theta, grad_outputs=torch.ones_like(u_theta), create_graph=True, retain_graph=True )[0]
        
        u_phi_phi = torch.autograd.grad(u_phi, self.phi, grad_outputs=torch.ones_like(u_phi), create_graph=True, retain_graph=True )[0]
        u_phi_phi_phi = torch.autograd.grad(u_phi_phi, self.phi, grad_outputs=torch.ones_like(u_phi), create_graph=True, retain_graph=True )[0] 
        
        div_uh = (1/(self.radius*torch.sin(self.theta)))*(u_theta_theta + u_phi_phi)
		
        #Calculating complexity
        u_theta_lap = ((1/self.radius**2 * torch.sin(self.theta)) *u_theta_theta_theta) + ((1/(self.radius * torch.sin(self.theta))**2)*u_theta_phi_phi)
        u_phi_lap = ((1/self.radius**2 * torch.sin(self.theta)) *u_phi_theta_theta) + ((1/(self.radius * torch.sin(self.theta))**2)*u_phi_phi_phi)
        comp = torch.sin(self.theta)*(u_theta_lap**2 + u_phi_lap**2)
        
        #  SV loss
        sv_out = -(u_theta*self.horiz_theta + u_phi*self.horiz_phi) - self.br*div_uh
        #TG assumption: div_H (u cos(theta) ) = 0; can also be written div_h u_h - tan(theta)/c u_theta = 0
        tg_con = div_uh - (u_theta*torch.tan(self.theta)/self.radius)

        return u_theta, u_phi, div_uh, sv_out, tg_con, comp
  
      	
class XavierInit(nn.Module):
    """
    Class to Xavier initialise weights so that the the varience of the activations are the same across all layers. 
    Taken from CEMAC PINNS notebook: https://github.com/cemac/LIFD_Torch_PINNS/tree/89151a8a9f0c161e53fcbc75d0c59a687fbbf5ad
    """
    def __init__(self, size):
        super().__init__()
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = torch.sqrt(torch.tensor(2.0 / (in_dim + out_dim)))
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * xavier_stddev)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

def initialize_NN(layers):
    """
    Initialising Neural Network from input layers. 
    Taken from CEMAC PINNS notebook: https://github.com/cemac/LIFD_Torch_PINNS/tree/89151a8a9f0c161e53fcbc75d0c59a687fbbf5ad
    """
    weights = nn.ModuleList()
    num_layers = len(layers)
    for l in range(num_layers - 1):
        layer = XavierInit(size=[layers[l], layers[l + 1]])
        weights.append(layer)
    return weights


def get_current_lr(optimizer, group_idx, parameter_idx):
    # Adam has different learning rates for each paramter. So we need to pick the
    # group and paramter first.
    group = optimizer.param_groups[group_idx]
    p = group['params'][parameter_idx]

    beta1, _ = group['betas']
    state = optimizer.state[p]

    bias_correction1 = 1 - beta1 ** state['step']
    current_lr = group['lr'] / bias_correction1
    return current_lr
    
class NeuralNet(nn.Module):
    """
    Class to define Pytorch NN from layers, lower bound and upper bound of coordinates.
    Taken from CEMAC PINNS notebook: https://github.com/cemac/LIFD_Torch_PINNS/tree/89151a8a9f0c161e53fcbc75d0c59a687fbbf5ad
    """
    def __init__(self, layers, lb, ub):
        super().__init__()
        self.weights = initialize_NN(layers)
        self.lb = torch.tensor(lb)
        self.ub = torch.tensor(ub)

    def forward(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(len(self.weights) - 1):
            H = torch.tanh(self.weights[l](H.float()))
        Y = self.weights[-1](H)
        return Y	
	
def train(nIter,model_name,radius_input, phi_input, theta_input, sv_input, horiz_theta_input, horiz_phi_input, br_input,  model, data_rec, flow_rec, loss_rec,  lr_rec,learning_rate=0.001, lamda_f = 1000):
    r"""
    Function to train model, producing horizontal flows at the CMB that 
    fit both the SV data and the flow assumptions. 

    Parameters
    ----------
    nITer : float, int
        Number of iterations for training. 
    model_name : str
        Path to file where model with the minimum loss will be saved. 
        No file extension needed, will be save in .pt format. 
        
    radius_input : torch.Tensor, ndarray
        Radial coordinates at the Core-Mantle Boundary, of shape (flattened(dx,dy), 1). 
        Units of km.
    theta_input : torch.Tensor, ndarray
        Latitude coordinates, of shape (flattened(dx,dy), 1). 
        Units of Radians.
    phi_input : torch.Tensor, ndarray
        Longitude coordinates, of shape (flattened(dx,dy), 1). 
        Units of Radians.    
	horiz_div_theta : torch.Tensor, ndarray
		Theta component of the horizontal divergence of the radial field component.
		Units of $\mu T/km$.
	horiz_div_phi : torch.Tensor, ndarray
		Phi component of the horizontal divergence of the radial field component.
		Units of $\mu T/km$.
    br_input : torch.Tensor, ndarray
		Radial magnetic field component.
        Units of $\mu T$.
        
    model : __main__.NeuralNet
        Initialised Pytorch Network, used to compute the Toroidal and Poloidal Scalars.
    data_rec : ndarray
   		Empty array to store SV Loss every iteration. Eg data_rec = np.zeros(Train_iterations).
	flow_rec : ndarray
   		Empty array to store Flow Assumption Loss every iteration. Eg flow_rec = np.zeros(Train_iterations).
   	loss_rec : ndarray
   		Empty array to store total Loss every iteration. Eg loss_rec = np.zeros(Train_iterations).
   		
    learning_rate : float, int, optional
        Step size at each iteration. 
        Defaults to 0.001.
    lamda :float, int, optional
        Weighting factor to ensure SV Loss and Flow loss vary on same scale. 
        Defaults to 1000.

	Notes
    -----
    Each term in the loss function weighted by np.sqrt(np.sin(theta)), to remove latitude 
    effects.
    """
    # Loss Criterion, currently Mean Squre Error
    criterion = nn.MSELoss()
    
    # Adam optimiser with default parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay = 1e-2)
    #This is updated as the model trains
    u_theta_pred, u_phi_pred, div_pred, sv_pred, tg_pred, comp = PINNFlow(radius_input, phi_input, theta_input, horiz_theta_input, horiz_phi_input, br_input,  model).net_sv()
    
    

       
    loss_data = criterion(sv_input*torch.sqrt(torch.sin(theta_input)), sv_pred*torch.sqrt(torch.sin(theta_input))) #SV LOSS
    loss_flows = lamda_f*criterion(tg_pred*(torch.sqrt(torch.sin(theta_input))), torch.zeros(tg_pred.shape)) #TG LOSS
    best_loss = loss_data + loss_flows
    start_time = time.time()
    
    for it in range(nIter):
        optimizer.zero_grad()
        #Calculating values at iteration it
        u_theta_pred, u_phi_pred, div_pred, sv_pred, tg_pred, comp = PINNFlow(radius_input, phi_input, theta_input, horiz_theta_input, horiz_phi_input, br_input,  model).net_sv()
        
        loss_data = criterion(sv_input*torch.sqrt(torch.sin(theta_input)), sv_pred*torch.sqrt(torch.sin(theta_input))) #SV LOSS
        loss_flows = lamda_f*criterion(tg_pred*(torch.sqrt(torch.sin(theta_input))), torch.zeros(tg_pred.shape)) #TG LOSS
        #loss_reg = 1000*criterion(comp*torch.sqrt(torch.sin(theta_input)), torch.zeros(comp.shape)) 
        loss = loss_data +loss_flows #+loss_reg #TOTAL LOSS
            
        #recording loss
        data_rec[it] = loss_data
        flow_rec[it] = loss_flows
        loss_rec[it] = loss
        #lr_rec[it] = optimizer.state_dict()['param_groups'][0]['lr']
        
	#Backpropagation
        loss.backward()
        optimizer.step()
        
        group_idx, param_idx = 0, 0
        current_lr = get_current_lr(optimizer, group_idx, param_idx)
        lr_rec[it] = current_lr
        
        # Print every 50 iterations
        if it % 50 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Total Loss: %.3e, SV Loss: %.3e, Flow Loss: %.3e, Time: %.2f' % 
                          (it, loss.item(), loss_data.item(), loss_flows.item(),  elapsed))
            start_time = time.time()
            #Checkpointing the 'Best Model', only saved if the loss is less than best_loss.
            if loss < best_loss:
                best_loss = loss
                torch.save(model, model_name+".pt")
                print("Loss Improved, Saving Model")
