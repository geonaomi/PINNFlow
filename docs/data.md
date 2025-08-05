# Loading in data

Function to compute:

* Radial Magnetic Field
* Secular Variation
* The (&theta;, &phi;) components of the horizontal divergence of the Radial Magnetic Field, 

using Gauss coefficients from `filename`. All evaluated on a `(dy,dx)` sized grid, defined by 
co-latitudes `(clat1,clat2)` and longitudes `(long1, long2)`. Each value is then rescaled so that all components in loss 
function are of order one. For more information on this, please see Notes.


```python
radius_data, phi_data, theta_data, Br_data, Br_dot_data, horiz_div_theta, horiz_div_phi = 
                data_generator(file_name, year, month =1, day =1, Nmax = 13, dy = 30, dx = 55, 
                clat1 = None, clat2 = None, long1 = None, long2 = None)
```

### Arguments

* `file_name` : *string* 
> Filepath and name of the MAT-file.

* `year` : *int, ndarray*

* `month` : *int, ndarray*, optional
> Defaults to 1 (January).

* `day` : *int, ndarray*, optional
> Defaults to 1.

* `Nmax` : *int*, positive
> Maximum Degree of the Spherical Harmonic Expansion, default 13.

* `dy` : *int*, positive, optional
> Number of grid points in the theta direction, default 30.

* `dx` : *int*, positive, optional
> Number of grid points in the phi direction, default 55.

* `clat1` : *ndarray, float*
> Colatitude, in degrees, of the upper boundary of the grid box.

* `clat2` : *ndarray, float*
> Colatitude, in degrees, of the lower boundary of the grid box.

* `long1` : *ndarray, float*
> Longitude, in degrees, of the left boundary of the grid box.

* `long2` : *ndarray, float*
> Longitude, in degrees, of the right boundary of the grid box.


### Returns

* `radius_data`, `theta_data`, `phi_data` : *ndarray*, shape `(dy, dx)`
> Radial, Theta, and Phi coordinates, each of shape (dx, dy).
> Theta and Phi coordinates given in radians. 

* `Br_data` : *ndarray*, shape `(dy, dx)`
> Radial field component, given in &mu;T

* `Br_data_dot` : *ndarray*, shape `(dy, dx)`
>Radial Secular Variation, given in &mu;T/0.1 years

* `horiz_div_theta` : *ndarray*, shape `(dy, dx)`
> Theta component of the horizontal divergence of the radial field component, given in &mu;T/km

* `horiz_div_phi` : *ndarray*, shape `(dy, dx)`
> Phi component of the horizontal divergence of the radial field component, given in &mu;T/km


### Notes


Re-scaling needed to ensure all inputs and outputs to the PINN are of the order one.
To do this:

* Br_data is rescaled from *nT* to *&mu;T*, which puts it at order ~1000

* Time is rescaled from year to 0.1 year

* Br_dot_data is rescaled from nT/year to &mu;T/0.1 years

* horiz_div_theta, horiz_div_phi are recaled from nT/m to &mu;T/km
	

### Source

