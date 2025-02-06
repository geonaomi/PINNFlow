# Loading in data

## Data Generator


Function to compute Radial Magnetic Field, Secular Variation, and the theta, phi components of the 
horizontal divergence of the Radial Magnetic Field, using Gauss coeffiecients from `filename`.

All evaluated on a grid `(dy,dx)` sized grid defined by 
co-latitudes `(clat1,clat2)` and longitudes `(long1, long2)`. 

Each value is then rescaled so that all components in loss 
function are of order one. For more information on this, please see Notes.


```python
data_generator(file_name, year, month =1, day =1, Nmax = 13, dy = 30, dx = 55, 
                clat1 = None, clat2 = None, long1 = None, long2 = None)
```

##### Arguments

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
