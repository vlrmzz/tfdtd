import numpy as np
import torch
import pytorch_lightning as pl 

import yaml
from Tfdtd.pml import PML

import logging
logging.basicConfig(filename='tensor_values.log', level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class TFDTD2D(pl.LightningModule):
    '''
    base class for 2D FDTD simulation with temporal 
    variant materials
    '''
    def __init__(self, config_file=None, params=None):
        super().__init__()
        
        if not (config_file or params):
            raise ValueError("You must provide either a config file or a parameter dictionary.")
        
        if config_file:
            print('Reading configuration from file...')
            self.params = self.read_config_file(config_file)
        else:
            self.params = params
            print('Reading configuration from dictionary...')
            
        self.geometries = []
        self.detectors = []
        self.sources = []

        self.initialize_params()
        self.initialize_grid()

        self.pml = PML(self.nx, 
                    self.ny, 
                    self.pml_thickness, 
                    backend=self.backend, 
                    precision=self.precision)

        self.calculate_e_field_coefficients()
        self.calculate_h_field_coefficients()
        self.update_fields = getattr(self, f"update_fields_{self.polarization.lower()}")
        if self.polarization == 'TE':
            self.update_fields = self.update_fields_te
        elif self.polarization == 'TM':
            self.update_fields = self.update_fields_tm
        else:
            raise ValueError(f"Invalid polarization: {self.polarization}. Must be 'TE' or 'TM'.")

    def read_config_file(self, config_file):
        with open(config_file, 'r') as file:
            params = yaml.safe_load(file)
        return params

    def initialize_params(self):
        # backend
        self.backend = self.params.get('backend', 'pytorch')
        self.precision = self.params.get('precision', 'float64')
        # Physical constants
        self.c = 299792458  # Speed of light in vacuum
        self.eps_0 = 8.85418782e-12  # Permittivity of free space
        self.mu_0 = 1.25663706e-6  # Permeability of free space
        self.mu_r = float(self.params.get('mu_r', 1.0)) # Relative permeability of air
        self.sigma = float(self.params.get('sigma_e', 0.0)) # Conductivity of air
        self.sigma_m = float(self.params.get('sigma_m', 0.0)) # Magnetic conductivity of air
        # Simulation parameters
        self.nx = int(self.params.get('nx', 60))
        self.ny = int(self.params.get('ny', 60))
        self.time_steps = int(self.params.get('time_steps', 50))
        self.time = 0.0
        self.dx = float(self.params.get('dx', 1e-9))
        self.dy = float(self.params.get('dy', 1e-9))
        self.dt = self.dx / (np.sqrt(2)*self.c)
        self.polarization = self.params.get('polarization', 'TM')
        self.use_pml = self.params.get('use_pml', True)
        self.pml_thickness = int(self.params.get('pml_thickness', 10))

    def init_pml_params(self):
        if self.use_pml:
            self.pml_thickness = int(self.params.get('pml_thickness', 20))
            self.pml = PML(self.nx, self.ny, self.pml_thickness)
        else:
            self.pml = None   

    # Init fields and coefficients
    def init_tensor_field(self, num_components):
        if self.backend not in ('numpy', 'pytorch'):
            raise ValueError("Invalid backend. Choose either 'numpy' or 'pytorch'.")
        if self.precision not in ('float32', 'float64'):
            raise ValueError("Invalid dtype. Choose either 'float32' or 'float64'.")

        if self.backend == 'numpy':
            return np.zeros((self.nx, self.ny, num_components), dtype=self.precision)
        elif self.backend == 'pytorch':
            return torch.zeros((self.nx, self.ny, num_components), dtype=getattr(torch, self.precision))


    def init_tensor_coefficient(self):
        if self.backend not in ('numpy', 'pytorch'):
            raise ValueError("Invalid backend. Choose either 'numpy' or 'pytorch'.")
        if self.precision not in ('float32', 'float64'):
            raise ValueError("Invalid dtype. Choose either 'float32' or 'float64'.")

        if self.backend == 'numpy':
            tensor = np.ones((self.nx, self.ny, self.time_steps + 1), dtype=self.precision)
            return tensor
        elif self.backend == 'pytorch':
            tensor = torch.ones((self.nx, self.ny, self.time_steps + 1), dtype=getattr(torch, self.precision))
            return tensor

    def initialize_grid(self):
        self.e_field = self.init_tensor_field(3)
        self.h_field = self.init_tensor_field(3)

        self.eps_r = self.init_tensor_coefficient()
        self.mu_r = self.init_tensor_coefficient() 

    def calculate_e_field_coefficients(self):
        """
        Calculate the electric field update coefficients ca and cb.

        Args:
            dt (float): Time step.
            eps_r (np.ndarray or torch.Tensor): Relative permittivity tensor with shape (nx, ny, nz).
            sigma (np.ndarray or torch.Tensor): Conductivity tensor with shape (nx, ny, nz).
            backend (str): Backend to use for creating tensors ('numpy' or 'pytorch').
            dtype (str): Data type for the tensors ('float32' or 'float64').

        Returns:
            ca (np.ndarray or torch.Tensor): Coefficient 'ca' tensor with shape (nx, ny, nz).
            cb (np.ndarray or torch.Tensor): Coefficient 'cb' tensor with shape (nx, ny, nz).
        """

        if self.backend == 'numpy':
            eaf = np.zeros((self.nx, self.ny, self.time_steps), dtype=self.precision)
            self.ca = np.zeros((self.nx, self.ny, self.time_steps), dtype=self.precision)
            self.cb = np.zeros((self.nx, self.ny, self.time_steps), dtype=self.precision)
        elif self.backend == 'pytorch':
            eaf = torch.zeros((self.nx, self.ny, self.time_steps), dtype=getattr(torch, self.precision))
            self.ca = torch.zeros((self.nx, self.ny, self.time_steps), dtype=getattr(torch, self.precision))
            self.cb = torch.zeros((self.nx, self.ny, self.time_steps), dtype=getattr(torch, self.precision))

        for t in range(self.time_steps):
            eaf[:, :, t] = self.dt * self.sigma / (2 * self.eps_0 * self.eps_r[:, :, t])
            self.ca[:, :, t] = (1 - eaf[:, :, t]) / (1 + eaf[:, :, t])
            self.cb[:, :, t] = 0.5 / (self.eps_r[:, :, t] * (1 + eaf[:, :, t]))


    def calculate_h_field_coefficients(self):
        """
        Calculate the magnetic field update coefficients da and db.

        Args:
            dt (float): Time step.
            mu_r (np.ndarray or torch.Tensor): Relative permeability tensor with shape (nx, ny, nz).
            sigma_m (np.ndarray or torch.Tensor): Magnetic conductivity tensor with shape (nx, ny, nz).
            backend (str): Backend to use for creating tensors ('numpy' or 'pytorch').
            dtype (str): Data type for the tensors ('float32' or 'float64').

        Returns:
            da (np.ndarray or torch.Tensor): Coefficient 'da' tensor with shape (nx, ny, nz).
            db (np.ndarray or torch.Tensor): Coefficient 'db' tensor with shape (nx, ny, nz).
        """

        if self.backend == 'numpy':
            eaf = np.zeros((self.nx, self.ny, self.time_steps), dtype=self.precision)
            self.da = np.zeros((self.nx, self.ny, self.time_steps), dtype=self.precision)
            self.db = np.zeros((self.nx, self.ny, self.time_steps), dtype=self.precision)
        elif self.backend == 'pytorch':
            eaf = torch.zeros((self.nx, self.ny, self.time_steps), dtype=getattr(torch, self.precision))
            self.da = torch.zeros((self.nx, self.ny, self.time_steps), dtype=getattr(torch, self.precision))
            self.db = torch.zeros((self.nx, self.ny, self.time_steps), dtype=getattr(torch, self.precision))

        for t in range(self.time_steps):
            eaf[:, :, t] = self.dt * self.sigma_m / (2 * self.mu_0 * self.mu_r[:, :, t])
            self.da[:, :, t] = (1 - eaf[:, :, t]) / (1 + eaf[:, :, t])
            self.db[:, :, t] = 0.5 / (self.mu_r[:, :, t] * (1 + eaf[:, :, t]))


    # Detectors
    def add_detector(self, detector):
        self.detectors.append(detector)

    def remove_detector(self, name):
        self.detectors = [detector for detector in self.detectors if detector.name != name]

    def record_detectors(self, field_values):
        for detector in self.detectors:
            detector.record(field_values)

    # Geometry
    def add_geometry(self, geometry):
        self.geometries.append(geometry)

    def remove_geometry(self, geometry):
        self.geometries.remove(geometry)
    
    def update_geometry(self):
        for geometry in self.geometries:
            mask = geometry.generate_mask(self.eps_r.shape[0], self.eps_r.shape[1])
            self.eps_r = self.eps_r * (1.0 - mask) + geometry.epsr * mask
            self.calculate_e_field_coefficients()
            self.calculate_h_field_coefficients()

    # Sources
    def add_source(self, source):
        self.sources.append(source)

    def remove_source(self, source):
        self.sources.remove(source)

    def update_source(self, time, dt, field):
        for source in self.sources:
            source.update_source(time, dt, field)

    # Fields updates
    def update_Ex_2d(self):
        """
        Update the electric field components for TE polarization in 2D using FDTD method.

        Args:
            e_field (np.ndarray or pytorch.tensor): Electric field tensor with shape (nx, ny, 2).
            h_field (np.ndarray or pytorch.tensor): Magnetic field tensor with shape (nx, ny, 1).
            ca (np.ndarray or pytorch.tensor): Coefficient 'ca' tensor with shape (nx, ny).
            cb (np.ndarray or pytorch.tensor): Coefficient 'cb' tensor with shape (nx, ny).


        Returns:
            None
        """
        # Update Ex component
        self.e_field[:-2, :-2, 0] = self.pml.gj3[:, :-2] * self.ca[:-2, :-2, self.actual_time_step] * self.e_field[:-2, :-2, 0] - \
                                    self.pml.gj2[:, :-2] * self.cb[:-2, :-2, self.actual_time_step] * (self.h_field[:-2, :-2, 2] - self.h_field[:-2, 1:-1, 2])

        return None

    def update_Ey_2d(self):
        """
        Update the electric field components for TE polarization in 2D using FDTD method.

        Args:
            e_field (np.ndarray or pytorch.tensor): Electric field tensor with shape (nx, ny, 2).
            h_field (np.ndarray or pytorch.tensor): Magnetic field tensor with shape (nx, ny, 1).
            ca (np.ndarray or pytorch.tensor): Coefficient 'ca' tensor with shape (nx, ny).
            cb (np.ndarray or pytorch.tensor): Coefficient 'cb' tensor with shape (nx, ny).

        Returns:
            None
        """

        # Update Ey component
        self.e_field[:-2, :-2, 1] = self.pml.gi3[:-2, :] * self.ca[:-2, :-2, self.actual_time_step] * self.e_field[:-2, :-2, 1] - \
                                    self.pml.gi2[:-2, :] * self.cb[:-2, :-2, self.actual_time_step] * (self.h_field[1:-1, :-2, 2] - self.h_field[:-2, :-2, 2])

        return None

    def update_Ez_2d(self):
        """
        Update the electric field components for TM polarization in 2D using FDTD method.

        Args:
            e_field (np.ndarray or pytorch.tensor): Electric field tensor with shape (nx, ny, 1).
            h_field (np.ndarray or pytorch.tensor): Magnetic field tensor with shape (nx, ny, 2).
            ca (np.ndarray or pytorch.tensor): Coefficient 'ca' tensor with shape (nx, ny).
            cb (np.ndarray or pytorch.tensor): Coefficient 'cb' tensor with shape (nx, ny).
        """
        self.e_field[1:-1, 1:-1, 2] = self.pml.gi3[1:-1, :] * self.pml.gj3[:, 1:-1] * self.ca[1:-1, 1:-1,self.actual_time_step]*self.e_field[1:-1, 1:-1, 2] + \
                    self.pml.gi2[1:-1, :] * self.pml.gj2[:, 1:-1]*self.cb[1:-1, 1:-1, self.actual_time_step] * ((self.h_field[1:-1, 1:-1, 1] - self.h_field[:-2, 1:-1, 1]) - \
                                    (self.h_field[1:-1, 1:-1, 0] - self.h_field[1:-1, :-2, 0]))

    def update_Hx_2d(self):
        """
        Update the magnetic field components for TE polarization in 2D using FDTD method.

        Args:
            h_field (np.ndarray or pytorch.tensor): Magnetic field tensor with shape (nx, ny, 2).
            e_field (np.ndarray or pytorch.tensor): Electric field tensor with shape (nx, ny, 1).
            da (np.ndarray or pytorch.tensor): Coefficient 'da' tensor with shape (nx, ny).
            db (np.ndarray or pytorch.tensor): Coefficient 'db' tensor with shape (nx, ny).

        Returns:
            None
        """
        # Update Hx component
        self.h_field[:-2, :-2, 0] =  self.pml.fj3[:, :-2] * self.da[:-2,:-2,self.actual_time_step] * self.h_field[:-2, :-2, 0] + \
                    self.pml.fj2[:, :-2] * self.db[:-2,:-2, self.actual_time_step] * (self.e_field[:-2, :-2, 2] - self.e_field[:-2, 1:-1, 2])

    def update_Hy_2d(self):
        """
        Update the magnetic field components for TE polarization in 2D using FDTD method.

        Args:
            h_field (np.ndarray or pytorch.tensor): Magnetic field tensor with shape (nx, ny, 2).
            e_field (np.ndarray or pytorch.tensor): Electric field tensor with shape (nx, ny, 1).
            da (np.ndarray or pytorch.tensor): Coefficient 'da' tensor with shape (nx, ny).
            db (np.ndarray or pytorch.tensor): Coefficient 'db' tensor with shape (nx, ny).

        Returns:
            None
        """
        # Update Hy component
        self.h_field[:-2, :-2, 1] = self.pml.fi3[:-2, :] * self.da[:-2,:-2,self.actual_time_step] * self.h_field[:-2, :-2, 1] + \
                            self.pml.fi2[:-2, :] * self.db[:-2,:-2, self.actual_time_step] * (self.e_field[1:-1, :-2, 2] - self.e_field[:-2, :-2, 2])


    def update_Hz_2d(self):
        """
        Update the magnetic field components for TM polarization in 2D using FDTD method.

        Args:
            h_field (np.ndarray or pytorch.tensor): Magnetic field tensor with shape (nx, ny, 1).
            e_field (np.ndarray or pytorch.tensor): Electric field tensor with shape (nx, ny, 2).
            da (np.ndarray or pytorch.tensor): Coefficient 'da' tensor with shape (nx, ny).
            db (np.ndarray or pytorch.tensor): Coefficient 'db' tensor with shape (nx, ny).

        Returns:
            None
        """
        # Update Hz component
        self.h_field[1:-1, 1:-1, 2] = self.pml.fj3[:, 1:-1] * self.pml.fi3[1:-1, :] * self.da[1:-1, 1:-1,self.actual_time_step] * self.h_field[1:-1, 1:-1, 2] + \
                self.pml.fj2[:, 1:-1] * self.pml.fi2[1:-1, :] * self.db[1:-1, 1:-1, self.actual_time_step] * ((self.e_field[1:-1, 1:-1, 0] - self.e_field[1:-1, :-2, 0]) -\
                                        (self.e_field[1:-1, 1:-1, 1] - self.e_field[:-2, 1:-1, 1]))

    # # Update TE
    def update_fields_te(self):
        self.update_Hz_2d()
        self.update_source(self.time, self.dt, self.h_field[:, :, 2])
        self.update_Ex_2d()
        self.update_Ey_2d()


    # Update TM
    def update_fields_tm(self):
        self.update_Ez_2d()
        self.update_source(self.time, self.dt, self.e_field[:, :, 2])
        self.update_Hx_2d()
        self.update_Hy_2d()
        
    def simulation_step(self, time_step):
        self.actual_time_step = time_step
        self.update_fields()
        self.time += self.dt