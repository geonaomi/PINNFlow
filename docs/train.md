# Training

Function that trains the model, producing horizontal flows at the Core Mantle Boundary (CMB) that fits both 
the SV and the TG flow assumption.

```python
train(nIter,model_name,radius_input, phi_input, theta_input, sv_input, horiz_theta_input, 
        horiz_phi_input, br_input,  model, data_rec, flow_rec, loss_rec, 
        learning_rate=0.001, lamda_f = 1000)
```

### Arguments

* `iIter` : *float, int*
> Number of iterations for training.

* `model_name` : *str*
> Path to file where the model will be saved. No file extension needed, will be saved in .pt format.

* `radius_input` : *torch.Tensor, ndarray*
> Radial coordinates at the Core-Mantle Boundary, of shape (flattened(dy,dx), 1). Units of km.

* `theta_input` : *torch.Tensor, ndarray*
> Latitude coordinates, of shape `(flattened(dy,dx), 1)`. Units of Radians.

* `phi_input` : *torch.Tensor, ndarray*
> Longitude coordinates, of shape `(flattened(dy,dx), 1)`. Units of Radians.

* `horiz_div_theta` : *torch.Tensor, ndarray*
> Theta component of the horizontal divergence of the radial field component. Units of &mu;T/km.

* `horiz_div_phi` : *torch.Tensor, ndarray*
> Phi component of the horizontal divergence of the radial field component. Units of &mu;T/km.

* `br_input` : *torch.Tensor, ndarray*
> Radial magnetic field component. Units of &mu;T/km.

* `model` : *__main__.NeuralNet*
> Initialised Pytorch Network, used to compute the Toroidal and Poloidal Scalars.

* `data_rec` : *ndarray*
> Empty array to store SV Loss every iteration. Eg `data_rec = np.zeros(Train_iterations)`.

* `flow_rec` : *ndarray*
> Empty array to store Flow Assumption Loss every iteration. Eg `flow_rec = np.zeros(Train_iterations)`.

* `loss_rec` : *ndarray*
> Empty array to store total Loss every iteration. Eg `loss_rec = np.zeros(Train_iterations)`.

* `learning_rate` : *float, int*, optional
> Step size at each iteration. Defaults to 0.001.

* `lamda` : *float, int* optional
> Weighting factor to ensure SV Loss and Flow Loss vary on same scale. Defaults to 1000.

### Notes

Each term in the loss function weighted by `np.sqrt(np.sin(theta))`, to remove latitude effects.

### Source