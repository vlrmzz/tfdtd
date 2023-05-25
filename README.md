# TFDTD: A Python library for 2D FDTD Simulations
TFDTD is a Python library that provides a framework for running 2D Finite-Difference Time-Domain (FDTD) simulations. 
The library is designed to work with both numpy and pytorch as computational backends, and supports time-varying and static materials. 
It provides classes and functions to define the simulation grid, electromagnetic field components, material properties, and update equations according to Maxwell's equations.

## Requirements
The TFDTD library requires the following Python packages:

numpy
pytorch
pytorch-lightning
yaml

## Installation
To use the TFDTD library, you can clone the repository to your local machine.

'''
git clone https://github.com/vlrmzz/tfdtd.git
'''

## How to use the TFDTD Library
The library primarily consists of two classes: FDTD2D and TFDTD2D.

TFDTD2D is a base class for 2D FDTD simulation with time-varying materials and FDTD2D is a class for 2D FDTD simulation with static materials.

The main simulation parameters and material properties are provided through a YAML configuration file or a Python dictionary. Parameters include grid size, time steps, cell dimensions, polarization, use of Perfectly Matched Layer (PML), PML thickness, and the backend computational library (either 'numpy' or 'pytorch').

In order to set up a simulation, you would first need to initialize a TFDTD2D or FDTD2D object with a configuration file or a parameters dictionary. Here is an example of how to initialize a FDTD2D object with a configuration file:

```
from tfdtd import FDTD2D

simulation = FDTD2D(config_file='path_to_config_file.yaml')

## Example: Simulation to find the modes of a waveguide
Here's a basic example of how you might set up a simulation to find the modes of a waveguide using this library. 
Note that this example is incomplete and requires further information about the waveguide geometry, material properties, and boundary conditions.


# define your simulation parameters in a dictionary or YAML file
params = {
    'backend': 'pytorch',
    'precision': 'float64',
    'mu_r': 1.0,
    'sigma_e': 0.0,
    'sigma_m': 0.0,
    'nx': 60,
    'ny': 60,
    'time_steps': 50,
    'dx': 1e-9,
    'dy': 1e-9,
    'polarization': 'TM',
    'use_pml': True,
    'pml_thickness': 10
}

# Initialize the FDTD2D object with the parameters
simulation = FDTD2D(params=params)

# Set up the waveguide geometry
waveguide_width = 10  # Width of the waveguide
waveguide_height = 5  # Height of the waveguide
waveguide_x_start = int(simulation.nx / 2 - waveguide_width / 2)  # X-coordinate of the waveguide start position
waveguide_x_end = int(simulation.nx / 2 + waveguide_width / 2)  # X-coordinate of the waveguide end position
waveguide_y_start = int(simulation.ny / 2 - waveguide_height / 2)  # Y-coordinate of the waveguide start position
waveguide_y_end = int(simulation.ny / 2 + waveguide_height / 2)  # Y-coordinate of the waveguide end position

# Set the waveguide material properties
simulation.eps_r[waveguide_x_start:waveguide_x_end, waveguide_y_start:waveguide_y_end] = waveguide_permittivity

# Setup the point source
source_params = {
    'x': 30, 
    'y': 30, 
    'amplitude': 1.0, 
    'function': 'gaussian', 
    'frequency_center': 3e8, 
    'frequency_width': 1e8
}
source = PointSource(source_params)
simulation.add_source(source)

# Run the simulation
for _ in range(simulation.time_steps):
    simulation.step()

# Access the electric and magnetic field values
e_field = simulation.e_field
h_field = simulation.h_field
```
