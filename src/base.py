import numpy as np
import torch
import pytorch_lightning as pl 

import yaml
from src.templates import simulation_step_template, point_source, plane_source, update_Dz_inc, set_boundaries
from src.templates import update_Ez_inc, update_Hx_inc, update_Hy_inc

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

        self.init_params()
        # Initialize time
        self.time = 0.0
        self.setup_simulation()

    def read_config_file(self, config_file):
        with open(config_file, 'r') as file:
            params = yaml.safe_load(file)
        return params

    def init_params(self):
        # Simulation parameters
        self.nx = int(self.params.get('nx', 60))
        self.ny = int(self.params.get('ny', 60))
        self.time_steps = int(self.params.get('time_steps', 50))
        self.time = 0.0
        self.dx = float(self.params.get('dx', 1e-9))
        self.dy = float(self.params.get('dy', 1e-9))
        self.dt = self.dx / 6e8
        # PML
        self.use_pml = self.params.get('use_pml', True)
        if self.use_pml == True:
            self.pml_thickness = int(self.params.get('pml_thickness', 20))
        # TFSF
        self.use_tfsf = self.params.get('use_tfsf', False)
        if self.use_tfsf == True:
            self.tfsf_thickness = int(self.params.get('tfsf_thickness', 10))
        self.polarization = self.params.get('polarization', 'TM')
        # Source
        self.function = self.params.get('function', 'gaussian')
        self.source_type = self.params.get('source_type', 'plane_wave')
        #plane_wave
        if self.source_type == 'plane_wave':
            self.frequency = float(self.params.get('frequency', 1e9))
            self.ia = int(self.params.get('plane_x1', 10))
            self.ja = int(self.params.get('plane_y1', 20))
            self.ib = int(self.params.get('plane_x2', 10))
            self.jb = int(self.params.get('plane_y2', 20))
        #point_source
        elif self.source_type == 'point_source':
            self.source_x = int(self.params.get('source_x', 150))
            self.source_y = int(self.params.get('source_y', 150))

        # Physical constants
        self.c = 299792458  # Speed of light in vacuum
        self.epsilon_0 = 8.85418782e-12  # Permittivity of free space
        self.mu_0 = 1.25663706e-6  # Permeability of free space

    def setup_simulation(self):
        # Initialize simulation parameters, grid, and fields
        self.init_params()
        self.init_grid()
        self.init_pml()
        if self.source_type == 'point_source':
            config_sim = simulation_step_template.format(
            update_Ez_inc = '',
            set_boundaries = '',
            update_Dz_inc = '',
            update_Hx_inc = '',
            update_Hy_inc = '',
            add_source= point_source,
            )
        elif self.source_type == 'plane_wave':
            self.init_plane_wave()
            config_sim = simulation_step_template.format(
            update_Ez_inc = update_Ez_inc,
            set_boundaries = set_boundaries,
            update_Dz_inc = update_Dz_inc,
            update_Hx_inc = update_Hx_inc,
            update_Hy_inc = update_Hy_inc,
            add_source= plane_source,
        )

        namespace = {}
        exec(config_sim, namespace)
        FDTD2D.simulation_step = namespace['simulation_step']
        

    def init_grid(self):
        # Initialize grid
        if self.polarization == 'TE':
            self.Ex = torch.zeros((self.nx, self.ny), dtype=torch.float64)
            self.Ey = torch.zeros((self.nx, self.ny), dtype=torch.float64)
            self.Hz = torch.zeros((self.nx, self.ny), dtype=torch.float64)
        elif self.polarization == 'TM':
            self.Hx = torch.zeros((self.nx, self.ny), dtype=torch.float64)
            self.Hy = torch.zeros((self.nx, self.ny), dtype=torch.float64)
            self.Ez = torch.zeros((self.nx, self.ny), dtype=torch.float64)
            self.Dz = torch.zeros((self.nx, self.ny), dtype=torch.float64)
            self.Iz = torch.zeros((self.nx, self.ny), dtype=torch.float64)
            self.iHx = torch.zeros((self.nx, self.ny), dtype=torch.float64)
            self.iHy = torch.zeros((self.nx, self.ny), dtype=torch.float64)
            self.gaz = torch.ones((self.nx, self.ny), dtype=torch.float64)
            self.gbz = torch.zeros((self.nx, self.ny), dtype=torch.float64)
        else:
            raise ValueError(f"Invalid polarization: {self.polarization}. Must be 'TE' or 'TM'.")

    def init_pml(self):
        i, j = self.nx, self.ny

        # Init coefficients for PML TM polarization
        if self.use_pml:
            self.gi2 = torch.ones((i,1))
            self.gi3 = torch.ones((i,1))
            self.fi1 = torch.zeros((i,1))
            self.fi2 = torch.ones((i,1))
            self.fi3 = torch.ones((i,1))

            self.gj2 = torch.ones((1,j))
            self.gj3 = torch.ones((1,j))
            self.fj1 = torch.zeros((1,j))
            self.fj2 = torch.ones((1,j))
            self.fj3 = torch.ones((1,j))

            # Create PML
            for n in range(self.pml_thickness):
                xnum = self.pml_thickness - n
                xd = self.pml_thickness
                xxn = xnum / xd
                xn = 0.33 * xxn ** 3

                self.gi2[n, 0] = 1 / (1 + xn)
                self.gi2[i - 1 - n, 0] = 1 / (1 + xn)
                self.gi3[n, 0] = (1 - xn) / (1 + xn)
                self.gi3[i - 1 - n, 0] = (1 - xn) / (1 + xn)

                self.gj2[0, n] = 1 / (1 + xn)
                self.gj2[0, j - 1 - n] = 1 / (1 + xn)
                self.gj3[0, n] = (1 - xn) / (1 + xn)
                self.gj3[0, j - 1 - n] = (1 - xn) / (1 + xn)

                xxn = (xnum - 0.5) / xd
                xn = 0.33 * xxn ** 3

                self.fi1[n, 0] = xn
                self.fi1[i - 2 - n, 0] = xn
                self.fi2[n, 0] = 1 / (1 + xn)
                self.fi2[i - 2 - n, 0] = 1 / (1 + xn)
                self.fi3[n, 0] = (1 - xn) / (1 + xn)
                self.fi3[i - 2 - n, 0] = (1 - xn) / (1 + xn)

                self.fj1[0, n] = xn
                self.fj1[0, j - 2 - n] = xn
                self.fj2[0, n] = 1 / (1 + xn)
                self.fj2[0, j - 2 - n] = 1 / (1 + xn)
                self.fj3[0, n] = (1 - xn) / (1 + xn)
                self.fj3[0, j - 2 - n] = (1 - xn) / (1 + xn)
    
    def init_plane_wave(self):
        self.Ez_inc = torch.zeros((1,self.ny), dtype=torch.float64)
        self.Hx_inc = torch.zeros((1,self.ny), dtype=torch.float64)
        #self.ib = self.nx - self.ia - 1 
        #self.jb = self.ny - self.ja - 1

        # Absorbing Boundary Conditions 
        self.boundary_low = [0, 0] 
        self.boundary_high = [0, 0]