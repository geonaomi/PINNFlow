# Defining PINNFlow Neural Network

Class that computes Toroidal and Poloidal Scalars, which are then differentiated to find flows.
Solutions should match both the input SV (sv_out = SV_CHAOS) and the flows constraint (tg_con = 0).

```python
class PINNFlow(nn.Module)
```

### Arguments

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

### Returns

All of the following are torch.Tensors of shape `(flattened(dy,dx), 1)`. To evaluate on a grid, use eg. `u_theta.flatten().detach().numpy()` to convert to an ndarray of shape `(flattened(dx,dy), 1)`, and re-shape to original grid shape using `np.reshape(...)`. 

* `u_theta` : *torch.Tensor*
> torch.Tensor representing the theta component of the flow.

* `u_phi` : *torch.Tensor*
> torch.Tensor representing the phi component of the flow. 

* `div_uh` : *torch.Tensor*
> torch.Tensor representing the horizontal divergence of the flow. 

* `sv_out` : *torch.Tensor*
> torch.Tensor representing the predicted SV from the calculated flows. 

* `tg_con` : *torch.Tensor*
> torch.Tensor representing the Tangentially Geostophic flow condition. 

* `comp` : *torch.Tensor*
> torch.Tensor representing the complexity, as defined by <a href="https://link.springer.com/chapter/10.1007/978-94-009-2857-2_9"> Bloxham (1988).</a> . 

# Supporting Utilities

The following classes and functions are used to build PINNflow, and are all taken from CEMAC PINNs example, which can be found <a href="https://github.com/cemac/LIFD_Torch_PINNS/tree/89151a8a9f0c161e53fcbc75d0c59a687fbbf5ad">here.</a> 

## Defining Blank Network

Class to define Pytorch NN from layers, lower bound and upper bound of coordinates. 

```python
class NeuralNet(nn.Module):
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
```

## Neural Network Initialisation

Initialising Neural Network from input layers. 

```python
def initialize_NN(layers):
    weights = nn.ModuleList()
        num_layers = len(layers)
        for l in range(num_layers - 1):
            layer = XavierInit(size=[layers[l], layers[l + 1]])
            weights.append(layer)
        return weights
```

## Xavier Initialisation

Class to Xavier initialise weights so that the the varience of the activations are the same across all layers.

```python
class XavierInit(nn.Module):
    def __init__(self, size):
        super().__init__()
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = torch.sqrt(torch.tensor(2.0 / (in_dim + out_dim)))
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * xavier_stddev)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias
```

