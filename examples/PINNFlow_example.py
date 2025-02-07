#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 09:32:02 2025

@author: Naomi Shakespeare-Rees
"""

#importing modules
import torch
import numpy as np
import matplotlib.pyplot as plt
from PINNFlow import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')

# Training on the GPU if it is available, CPU if not. 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Path to MAT-File
mat_file = "CHAOS-8.1.mat"

#Loading in data from CHAOS, 1st Jan 2024, degree 13, on a (30,55) grid over the Atlantic. 
#Change clat and long variables if you would like to change the region, (dy,dx)  if you would like to change the number of points.
radius_data, phi_data, theta_data, Br_data, Br_dot_data, Br_div_theta_data , Br_div_phi_data = data_generator(mat_file, 2024, 1, 1, 13,  dy = 30 , dx = 55, clat1 = 55,clat2 = 85, long1 = -50, long2 = 5)


#Plotting input data to ensure it is correct
plt.rcParams.update({'font.size': 16})
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(15,10))
plot1 = ax1.scatter(phi_data*180/np.pi, 90-theta_data*180/np.pi,c=Br_data,cmap = 'bwr')
plot2 = ax2.scatter(phi_data*180/np.pi, 90-theta_data*180/np.pi, c=Br_dot_data,cmap = 'bwr')
plt.colorbar(plot1, ax = ax1, label = '$\mu$ T')
plt.colorbar(plot2, ax = ax2, label = '$\mu$ T/0.1yr')
plt.tight_layout(pad=3)
ax1.set_title('Magnetic field (Br) \n')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax2.set_title('Rate of change of the magnetic field SV (Br_dot) \n')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
plot3 = ax3.scatter(phi_data*180/np.pi, 90-theta_data*180/np.pi, c=Br_div_theta_data,cmap = 'bwr')
plot4 = ax4.scatter(phi_data*180/np.pi, 90-theta_data*180/np.pi, c=Br_div_phi_data,cmap = 'bwr')
plt.colorbar(plot3, ax = ax3, label = '$\mu$ T/km')
plt.colorbar(plot4, ax = ax4, label = '$\mu$ T/km')
plt.tight_layout(pad=3)
ax3.set_title('Derivative of Br with respect to latitude \n')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax4.set_title('Derivative of Br with respect to longitude \n')
ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')


#Pytorch wants inputs flattened 
theta = theta_data.flatten()[:,None]
phi = phi_data.flatten()[:,None]
radius = 3485*np.ones(theta.shape)
print("theta shape: ", theta.shape, ",phi shape: ", phi.shape)
print("sv shape: ", Br_dot_data.shape)
PHI,THETA = np.meshgrid(phi, theta)

coord =  np.hstack((phi.flatten()[:,None], theta.flatten()[:,None])) 
sv_star = Br_dot_data.flatten()[:, None]
br_star = Br_data.flatten()[:, None]
horiz_theta_star = Br_div_theta_data.flatten()[:, None]
horiz_phi_star = Br_div_phi_data.flatten()[:, None]
print("Coord shape: ", coord.shape) #Needed to initialise network
print("SV_STAR shape: ", sv_star.shape)

#Convert to torch.Tensor 
radius_tf = torch.tensor(radius, requires_grad=True).float()
theta_tf = torch.tensor(theta, requires_grad=True).float()
phi_tf = torch.tensor(phi, requires_grad=True).float()
br_tf = torch.tensor(br_star, requires_grad=True).float()
sv_tf = torch.tensor(sv_star, requires_grad=True).float()
horiz_theta_tf = torch.tensor(horiz_theta_star, requires_grad=True).float()
horiz_phi_tf = torch.tensor(horiz_phi_star, requires_grad=True).float()

#Default Network Size, can be changed but start and end values need to be 2.
layers =[2,40, 40, 40, 40, 40, 40, 40, 40, 2]
lb = coord.min(0) #Lower Bound for coordinate system 
ub = coord.max(0) #Upper Bound for coordinate system 
model = NeuralNet(layers, lb, ub) #Defining NN model for T, P

Train_iterations = 10_000#Number of training iterations

# Calling Training function and timing it 
start = time.time()

#Defining empty arrays to store loss values
loss_record = np.zeros(Train_iterations)
loss_data_record = np.zeros(Train_iterations)
loss_flows_record = np.zeros(Train_iterations)
#Training, will automatically save the best performing model in "PINNFlow_model.pt"
train(Train_iterations, "PINNFlow_model", radius_tf, phi_tf, theta_tf, sv_tf, horiz_theta_tf, horiz_phi_tf,br_tf,  model, loss_data_record, loss_flows_record, loss_record, lamda_f = 1000)
end = time.time()

time_taken = end-start
print("Time Taken for Training {0:.2f} minutes.".format(time_taken/60))

#Plotting loss curves to ensure convergence 
plt.rcParams.update({'font.size': 18})
fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(25,10))
ax1.semilogy(loss_data_record, color = 'blue')
ax2.semilogy(loss_flows_record, color = 'red')
ax3.semilogy(loss_record, color = 'green')
ax1.set_title("SV Loss")
ax2.set_title("TG Loss")

ax3.set_title("Total Loss")
fig.supxlabel('Iteration')
fig.supylabel('Loss')
fig.suptitle('Loss \n', fontsize = 35)
plt.tight_layout(pad=1)


model = torch.load("PINNFlow_model.pt", weights_only=False) #Loading in 'Best Model'

#evaluating trained model values
u_theta_flat, u_phi_flat, div_flat, sv_flat, tg_flat, cond = PINNFlow(radius_tf, phi_tf, theta_tf,  horiz_theta_tf, horiz_phi_tf,br_tf, model).net_sv()
#detaching from cuda, converting to numpy, reshaping to original grid and removing 5 degree border
u_theta = -u_theta_flat.flatten().detach().numpy().reshape(Br_dot_data.shape)[5:-5, 5:-5]*10
u_phi = u_phi_flat.flatten().detach().numpy().reshape(Br_dot_data.shape)[5:-5, 5:-5]*10
sv_pred = sv_flat.flatten().detach().numpy().reshape(Br_dot_data.shape)[5:-5, 5:-5]
tg_pred = tg_flat.flatten().detach().numpy().reshape(Br_dot_data.shape)[5:-5, 5:-5]
div_pred = div_flat.flatten().detach().numpy().reshape(Br_dot_data.shape)[5:-5, 5:-5]

#converting to nT/year
sv_nt = Br_dot_data[5:-5, 5:-5]*1e4
sv_pred_nt = (sv_pred*1e4)#/np.sin(theta_data[5:-5, 5:-5])
#calculating residual
res = sv_nt-sv_pred_nt
rmse_sv = np.sqrt(mean_squared_error(sv_nt, sv_pred_nt))
print("RMSE between CHAOS SV and Predicted SV = {0:.0f} nT/year.".format(rmse_sv))
permse = (rmse_sv*100)/np.abs(sv_nt).max()
print("Percentage RMSE between CHAOS SV and Predicted SV = {0:.2f}%.".format(permse))

#Plotting results
fig = plt.figure(figsize=(20, 20))
proj = ccrs.Mollweide() 
ax = plt.axes(projection=proj)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.COASTLINE)
plot2 = ax.quiver(phi_data[5:-5, 5:-5]*180/np.pi, 90-theta_data[5:-5, 5:-5]*180/np.pi, u_phi ,u_theta,color = 'black', transform=ccrs.PlateCarree())
ax.quiverkey(plot2, 0.8, 0.9, 20, label = '20 km/yr', labelpos='W',coordinates='figure',  fontproperties={'size':30})
ax.set_title("Recovered Flows \n", size = '30')

#PLotting Recovered SV
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(30,10))
plot1 = ax1.scatter(phi_data[5:-5, 5:-5]*180/np.pi, 90-theta_data[5:-5, 5:-5]*180/np.pi, c=Br_dot_data[5:-5, 5:-5]*1e4,cmap = 'bwr', s=200)
plot3 =ax2.scatter(phi_data[5:-5, 5:-5]*180/np.pi, 90-theta_data[5:-5, 5:-5]*180/np.pi, c=sv_pred_nt,cmap = 'bwr', s=200)
plot4 =ax3.scatter(phi_data[5:-5, 5:-5]*180/np.pi, 90-theta_data[5:-5, 5:-5]*180/np.pi, c=res,cmap = 'bwr', vmin = -np.abs(res).max(), vmax = np.abs(res).max(), s=200)
plt.colorbar(plot1, ax = ax1, label = 'nT/year')
plt.colorbar(plot3, ax = ax2, label = 'nT/year')
plt.colorbar(plot4, ax = ax3, label = 'nT/year')
plt.tight_layout(pad=3)
ax1.set_title('"True" SV', fontsize = 25)
ax1.set_xlabel('Longitude', fontsize = 15)
ax1.set_ylabel('Latitude', fontsize = 15)

ax2.set_title('Output SV', fontsize = 25)
ax2.set_xlabel('Longitude', fontsize = 15)
ax2.set_ylabel('Latitude', fontsize = 15)

ax4.set_title('Residual SV, RMSE {0:.0f} nT/year.'.format(rmse_sv), fontsize = 25)
ax4.set_xlabel('Longitude', fontsize = 15)
ax4.set_ylabel('Latitude', fontsize = 15)
plt.suptitle("Input and Output SV, TG Assumption", fontsize = 40, y = 1.05)

