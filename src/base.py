import numpy as np
import torch
import pytorch_lightning as pl 

import yaml
from src.pml import PML

import logging
logging.basicConfig(filename='tensor_values.log', level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class FDTD2D(pl.LightningModule):
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
        self.calculate_e_field_coefficients()
        self.calculate_h_field_coefficients()
        self.initialize_source()
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
        # Physical constants
        self.c = 299792458  # Speed of light in vacuum
        self.eps_0 = 8.85418782e-12  # Permittivity of free space
        self.mu_0 = 1.25663706e-6  # Permeability of free space
        self.mu_r = 1.0 # Relative permeability of air
        self.sigma = 0.0 # Conductivity of air
        self.sigma_m = 0.0 # Magnetic conductivity of air

        self.nx = int(self.params.get('nx', 60))
        self.ny = int(self.params.get('ny', 60))
        self.time_steps = int(self.params.get('time_steps', 50))
        self.time = 0.0
        self.dx = float(self.params.get('dx', 1e-9))
        self.dy = float(self.params.get('dy', 1e-9))
        self.dt = self.dx / (np.sqrt(2)*self.c)
        self.polarization = self.params.get('polarization', 'TM')
        # setup to get te follwing from the config file
        self.backend = 'pytorch'
        self.precision = 'float64'

    def init_pml_params(self):
        self.use_pml = self.params.get('use_pml', True)
        if self.use_pml:
            self.pml_thickness = int(self.params.get('pml_thickness', 20))
            self.pml = PML(self.nx, self.ny, self.pml_thickness)
        else:
            self.pml = None

    def initialize_source(self):
        self.function = self.params.get('function', 'gaussian')
        if self.function == 'sinusoidal':
            self.frequency = float(self.params.get('frequency', 1e9))
        self.source_type = self.params.get('source_type', 'point_source')
        if self.source_type == 'line_source':
            self.line_x = int(self.params.get('line_x', 10))
            self.line_y1 = int(self.params.get('line_y1', 20))
            self.line_y2 = int(self.params.get('line_y2', 20))
        elif self.source_type == 'point_source':
            self.source_x = int(self.params.get('source_x', 150))
            self.source_y = int(self.params.get('source_y', 150))      

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
            return np.zeros((self.nx, self.ny), dtype=self.precision)
        elif self.backend == 'pytorch':
            return torch.zeros((self.nx, self.ny), dtype=getattr(torch, self.precision))

    def initialize_grid(self):
        self.e_field = self.init_tensor_field(3)
        self.h_field = self.init_tensor_field(3)

        self.eps_r = self.init_tensor_coefficient() + 1.0
        self.mu_r = self.init_tensor_coefficient() + 1.0

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

        #self.ca = (1 - (self.sigma * self.dt) / (2 * self.eps_0 * self.eps_r)) / (1 + (self.sigma * self.dt) / (2 * self.eps_0 * self.eps_r))
        #self.cb = (self.dt / (self.eps_0 * self.eps_r * self.dx )) / (1 + (self.sigma * self.dt)/(2 * self.eps_0 * self.eps_r))
        eaf = self.dt * self.sigma /(2 * self.eps_0 * self.eps_r)
        self.ca = (1 - eaf)/ (1 + eaf)
        self.cb = 0.5/(self.eps_r*( 1 + eaf))

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

        #self.da = (1 - (self.sigma_m * self.dt) / (2 * self.mu_0 * self.mu_r)) / (1 + (self.sigma_m * self.dt) / (2 * self.mu_0 * self.mu_r))
        #self.db = (self.dt / (self.mu_0 * self.mu_r * self.dx )) / (1 + (self.sigma_m * self.dt) / (2 * self.mu_0 * self.mu_r))
        eaf = self.dt * self.sigma_m /(2 * self.mu_0 * self.mu_r)
        self.da = (1 - eaf)/ (1 + eaf)
        self.db = 0.5/(self.mu_r*( 1 + eaf))

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
            dx (float): Spatial step along the x-axis.
            dy (float): Spatial step along the y-axis.

        Returns:
            None
        """
        # Update Ex component
        self.e_field[:-2, :-2, 0] = self.ca[:-2, :-2] * self.e_field[:-2, :-2, 0] - \
                                    self.cb[:-2, :-2] * (self.h_field[:-2, :-2, 2] - self.h_field[:-2, 1:-1, 2])

        return None

    def update_Ey_2d(self):
        """
        Update the electric field components for TE polarization in 2D using FDTD method.

        Args:
            e_field (np.ndarray or pytorch.tensor): Electric field tensor with shape (nx, ny, 2).
            h_field (np.ndarray or pytorch.tensor): Magnetic field tensor with shape (nx, ny, 1).
            ca (np.ndarray or pytorch.tensor): Coefficient 'ca' tensor with shape (nx, ny).
            cb (np.ndarray or pytorch.tensor): Coefficient 'cb' tensor with shape (nx, ny).
            dx (float): Spatial step along the x-axis.
            dy (float): Spatial step along the y-axis.

        Returns:
            None
        """

        # Update Ey component
        self.e_field[:-2, :-2, 1] = self.ca[:-2, :-2] * self.e_field[:-2, :-2, 1] - \
                                    self.cb[:-2, :-2] * (self.h_field[1:-1, :-2, 2] - self.h_field[:-2, :-2, 2])

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
        self.e_field[1:-1, 1:-1, 2] = self.ca[1:-1, 1:-1]*self.e_field[1:-1, 1:-1, 2] + \
                    self.cb[1:-1, 1:-1] * ((self.h_field[1:-1, 1:-1, 1] - self.h_field[:-2, 1:-1, 1]) - \
                                    (self.h_field[1:-1, 1:-1, 0] - self.h_field[1:-1, :-2, 0]))

    def update_Hx_2d(self):
        """
        Update the magnetic field components for TE polarization in 2D using FDTD method.

        Args:
            h_field (np.ndarray or pytorch.tensor): Magnetic field tensor with shape (nx, ny, 2).
            e_field (np.ndarray or pytorch.tensor): Electric field tensor with shape (nx, ny, 1).
            da (np.ndarray or pytorch.tensor): Coefficient 'da' tensor with shape (nx, ny).
            db (np.ndarray or pytorch.tensor): Coefficient 'db' tensor with shape (nx, ny).
            dx (float): Spatial step along the x-axis.
            dy (float): Spatial step along the y-axis.

        Returns:
            None
        """
        # Update Hx component
        self.h_field[:-2, :-2, 0] =  self.da[:-2,:-2] * self.h_field[:-2, :-2, 0] + \
                    self.db[:-2,:-2] * (self.e_field[:-2, :-2, 2] - self.e_field[:-2, 1:-1, 2])

    def update_Hy_2d(self):
        """
        Update the magnetic field components for TE polarization in 2D using FDTD method.

        Args:
            h_field (np.ndarray or pytorch.tensor): Magnetic field tensor with shape (nx, ny, 2).
            e_field (np.ndarray or pytorch.tensor): Electric field tensor with shape (nx, ny, 1).
            da (np.ndarray or pytorch.tensor): Coefficient 'da' tensor with shape (nx, ny).
            db (np.ndarray or pytorch.tensor): Coefficient 'db' tensor with shape (nx, ny).
            dx (float): Spatial step along the x-axis.
            dy (float): Spatial step along the y-axis.

        Returns:
            None
        """
        # Update Hy component
        self.h_field[:-2, :-2, 1] = self.da[:-2,:-2] * self.h_field[:-2, :-2, 1] + \
                            self.db[:-2,:-2] * (self.e_field[1:-1, :-2, 2] - self.e_field[:-2, :-2, 2])


    def update_Hz_2d(self):
        """
        Update the magnetic field components for TM polarization in 2D using FDTD method.

        Args:
            h_field (np.ndarray or pytorch.tensor): Magnetic field tensor with shape (nx, ny, 1).
            e_field (np.ndarray or pytorch.tensor): Electric field tensor with shape (nx, ny, 2).
            da (np.ndarray or pytorch.tensor): Coefficient 'da' tensor with shape (nx, ny).
            db (np.ndarray or pytorch.tensor): Coefficient 'db' tensor with shape (nx, ny).
            dx (float): Spatial step along the x-axis.
            dy (float): Spatial step along the y-axis.

        Returns:
            None
        """
        # Update Hz component
        self.h_field[1:-1, 1:-1, 2] = self.da[1:-1, 1:-1] * self.h_field[1:-1, 1:-1, 2] + \
                self.db[1:-1, 1:-1] * ((self.e_field[1:-1, 1:-1, 0] - self.e_field[1:-1, :-2, 0]) -\
                                        (self.e_field[1:-1, 1:-1, 1] - self.e_field[:-2, 1:-1, 1]))

    # # Update TE
    def update_fields_te(self):
        self.update_Hz_2d()
        self.update_source(self.time, self.dt, self.h_field[:, :, 2])
        self.update_Ex_2d()
        #self.update_source(self.time, self.dt, self.e_field[:, :, 0])
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
        # Update time
        self.time += self.dt