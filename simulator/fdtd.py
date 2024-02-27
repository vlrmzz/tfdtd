import numpy as np
import torch
import pytorch_lightning as pl 

import yaml
from simulator.pml import PML
from simulator.base import TFDTD2D

import logging
logging.basicConfig(filename='tensor_values.log', level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class FDTD2D(TFDTD2D):
    '''
    class for 2D FDTD simulation with static materials
    '''
    # overrride initalization of the coeffcients 
    def init_tensor_coefficient(self):
        if self.backend not in ('numpy', 'pytorch'):
            raise ValueError("Invalid backend. Choose either 'numpy' or 'pytorch'.")
        if self.precision not in ('float32', 'float64'):
            raise ValueError("Invalid dtype. Choose either 'float32' or 'float64'.")

        if self.backend == 'numpy':
            return np.ones((self.nx, self.ny), dtype=self.precision)
        elif self.backend == 'pytorch':
            return torch.ones((self.nx, self.ny), dtype=getattr(torch, self.precision))

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
            eaf = np.zeros((self.nx, self.ny), dtype=self.precision)
            self.ca = np.zeros((self.nx, self.ny), dtype=self.precision)
            self.cb = np.zeros((self.nx, self.ny), dtype=self.precision)
        elif self.backend == 'pytorch':
            eaf = torch.zeros((self.nx, self.ny), dtype=getattr(torch, self.precision))
            self.ca = torch.zeros((self.nx, self.ny), dtype=getattr(torch, self.precision))
            self.cb = torch.zeros((self.nx, self.ny), dtype=getattr(torch, self.precision))

    
        eaf[:, :] = self.dt * self.sigma / (2 * self.eps_0 * self.eps_r[:, :])
        self.ca[:, :] = (1 - eaf[:, :]) / (1 + eaf[:, :])
        self.cb[:, :] = 0.5 / (self.eps_r[:, :] * (1 + eaf[:, :]))


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
            eaf = np.zeros((self.nx, self.ny), dtype=self.precision)
            self.da = np.zeros((self.nx, self.ny), dtype=self.precision)
            self.db = np.zeros((self.nx, self.ny), dtype=self.precision)
        elif self.backend == 'pytorch':
            eaf = torch.zeros((self.nx, self.ny), dtype=getattr(torch, self.precision))
            self.da = torch.zeros((self.nx, self.ny), dtype=getattr(torch, self.precision))
            self.db = torch.zeros((self.nx, self.ny), dtype=getattr(torch, self.precision))

        
        eaf[:, :] = self.dt * self.sigma_m / (2 * self.mu_0 * self.mu_r[:, :])
        self.da[:, :] = (1 - eaf[:, :]) / (1 + eaf[:, :])
        self.db[:, :] = 0.5 / (self.mu_r[:, :] * (1 + eaf[:, :]))

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
        self.e_field[:-2, :-2, 0] = self.pml.gj3[:, :-2] * self.ca[:-2, :-2] * self.e_field[:-2, :-2, 0] - \
                                    self.pml.gj2[:, :-2] * self.cb[:-2, :-2] * (self.h_field[:-2, :-2, 2] - self.h_field[:-2, 1:-1, 2])

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
        self.e_field[:-2, :-2, 1] = self.pml.gi3[:-2, :] * self.ca[:-2, :-2] * self.e_field[:-2, :-2, 1] - \
                                    self.pml.gi2[:-2, :] * self.cb[:-2, :-2] * (self.h_field[1:-1, :-2, 2] - self.h_field[:-2, :-2, 2])

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
        self.e_field[1:-1, 1:-1, 2] = self.pml.gi3[1:-1, :] * self.pml.gj3[:, 1:-1] * self.ca[1:-1, 1:-1]*self.e_field[1:-1, 1:-1, 2] + \
                    self.pml.gi2[1:-1, :] * self.pml.gj2[:, 1:-1]*self.cb[1:-1, 1:-1] * ((self.h_field[1:-1, 1:-1, 1] - self.h_field[:-2, 1:-1, 1]) - \
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
        self.h_field[:-2, :-2, 0] =  self.pml.fj3[:, :-2] * self.da[:-2,:-2] * self.h_field[:-2, :-2, 0] + \
                    self.pml.fj2[:, :-2] * self.db[:-2,:-2] * (self.e_field[:-2, :-2, 2] - self.e_field[:-2, 1:-1, 2])

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
        self.h_field[:-2, :-2, 1] = self.pml.fi3[:-2, :] * self.da[:-2,:-2] * self.h_field[:-2, :-2, 1] + \
                            self.pml.fi2[:-2, :] * self.db[:-2,:-2] * (self.e_field[1:-1, :-2, 2] - self.e_field[:-2, :-2, 2])


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
        self.h_field[1:-1, 1:-1, 2] = self.pml.fj3[:, 1:-1] * self.pml.fi3[1:-1, :] * self.da[1:-1, 1:-1] * self.h_field[1:-1, 1:-1, 2] + \
                self.pml.fj2[:, 1:-1] * self.pml.fi2[1:-1, :] * self.db[1:-1, 1:-1] * ((self.e_field[1:-1, 1:-1, 0] - self.e_field[1:-1, :-2, 0]) -\
                                        (self.e_field[1:-1, 1:-1, 1] - self.e_field[:-2, 1:-1, 1]))