import numpy as np
import torch
import pytorch_lightning as pl 

import yaml
from src.pml import PML


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

        self.init_params()
        self.init_grid()
        self.update_fields = getattr(self, f"update_fields_{self.polarization.lower()}")
        # if self.polarization == 'TE':
        #     self.update_fields = self.update_fields_te
        # elif self.polarization == 'TM':
        #     self.update_fields = self.update_fields_tm
        # else:
        #     raise ValueError(f"Invalid polarization: {self.polarization}. Must be 'TE' or 'TM'.")

    def read_config_file(self, config_file):
        with open(config_file, 'r') as file:
            params = yaml.safe_load(file)
        return params

    def init_params(self):
        self.init_simulation_params()
        self.init_pml_params()
        self.init_tfsf_params()
        self.init_source_params()

    def init_simulation_params(self):
        # Physical constants
        self.c = 299792458  # Speed of light in vacuum
        self.epsilon_0 = 8.85418782e-12  # Permittivity of free space
        self.mu_0 = 1.25663706e-6  # Permeability of free space

        self.nx = int(self.params.get('nx', 60))
        self.ny = int(self.params.get('ny', 60))
        self.time_steps = int(self.params.get('time_steps', 50))
        self.time = 0.0
        self.dx = float(self.params.get('dx', 1e-9))
        self.dy = float(self.params.get('dy', 1e-9))
        self.dt = self.dx / self.c
        self.polarization = self.params.get('polarization', 'TM')

    def init_pml_params(self):
        self.use_pml = self.params.get('use_pml', True)
        if self.use_pml:
            self.pml_thickness = int(self.params.get('pml_thickness', 20))
            self.pml = PML(self.nx, self.ny, self.pml_thickness)
        else:
            self.pml = None

    def init_tfsf_params(self):
        self.use_tfsf = self.params.get('use_tfsf', False)
        if self.use_tfsf:
            self.tfsf_thickness = int(self.params.get('tfsf_thickness', 10))

    def init_source_params(self):
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
    
    def init_grid(self):
        if self.polarization == 'TE':
            self.init_te_fields()
        elif self.polarization == 'TM':
            self.init_tm_fields()
        else:
            raise ValueError(f"Invalid polarization: {self.polarization}. Must be 'TE' or 'TM'.")

    def init_te_fields(self):
        fields = ['Ex', 'Ey', 'Hz', 'Dx', 'Dy', 'Ix', 'Iy', 'iHz', 'gbx', 'gby']
        self.gax = torch.ones((self.nx, self.ny), dtype=torch.float64)
        self.gay = torch.ones((self.nx, self.ny), dtype=torch.float64)
        for field in fields:
            setattr(self, field, torch.zeros((self.nx, self.ny), dtype=torch.float64))

    def init_tm_fields(self):
        fields = ['Hx', 'Hy', 'Ez', 'Dz', 'Iz', 'iHx', 'iHy', 'gbz']
        self.gaz = torch.ones((self.nx, self.ny), dtype=torch.float64)
        for field in fields:
            setattr(self, field, torch.zeros((self.nx, self.ny), dtype=torch.float64))        

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

    def update_grid(self):
        if self.polarization == 'TE':
            for geometry in self.geometries:
                mask = geometry.generate_mask(self.nx, self.ny)
                self.gax[0:self.nx, 0:self.ny] = mask * (1 / (geometry.epsr + (geometry.sigma * self.dt / self.epsilon_0))) \
                                                + (1 - mask) * self.gax[0:self.nx, 0:self.ny]
                self.gbx[0:self.nx, 0:self.ny] = mask * ((geometry.sigma * self.dt / self.epsilon_0))
                
                self.gay[0:self.nx, 0:self.ny] = mask * (1 / (geometry.epsr + (geometry.sigma * self.dt / self.epsilon_0))) \
                                                + (1 - mask) * self.gay[0:self.nx, 0:self.ny]
                self.gby[0:self.nx, 0:self.ny] = mask * ((geometry.sigma * self.dt / self.epsilon_0))
        elif self.polarization == 'TM':
            for geometry in self.geometries:
                print(geometry.epsr)
                mask = geometry.generate_mask(self.nx, self.ny)
                self.gaz[0:self.nx, 0:self.ny] = mask * (1 / (geometry.epsr + (geometry.sigma * self.dt / self.epsilon_0))) \
                                             + (1 - mask) * self.gaz[0:self.nx, 0:self.ny]
                self.gbz[0:self.nx, 0:self.ny] = mask * ((geometry.sigma * self.dt / self.epsilon_0)) 

    # Sources
    def add_source(self, source):
        self.sources.append(source)

    def remove_source(self, source):
        self.sources.remove(source)

    def update_source(self, time, dt, field):
        for source in self.sources:
            source.update_source(time, dt, field)

    # Fields updates
    def update_Dx(self):
        ie, je = self.nx, self.ny

        # Calculate Dx
        self.Dx[1:ie, 1:je] = self.pml.gi3[1:ie, :] * self.pml.gj3[:, 1:je] * self.Dx[1:ie, 1:je] - \
                              self.pml.gi2[1:ie, :] * self.pml.gj2[:, 1:je] * 0.5 * (self.Hz[1:ie, 1:je] - self.Hz[1:ie, 0:je-1])

    def update_Dy(self):
        ie, je = self.nx, self.ny

        # Calculate Dy
        self.Dy[1:ie, 1:je] = self.pml.gi3[1:ie, :] * self.pml.gj3[:, 1:je] * self.Dy[1:ie, 1:je] + \
                              self.pml.gi2[1:ie, :] * self.pml.gj2[:, 1:je] * 0.5 * (self.Hz[1:ie, 1:je] - self.Hz[0:ie-1, 1:je])

    def update_Ex(self):
        # Calculate Ex
        self.Ex = self.gax * (self.Dx - self.Ix)
        self.Ix = self.Ix + self.gbx * self.Ex

    def update_Ey(self):
        # Calculate Ey
        self.Ey = self.gay * (self.Dy - self.Iy)
        self.Iy = self.Iy + self.gby * self.Ey


    def update_Dz(self):
        ie, je = self.nx, self.ny

        # Calculate Dz
        self.Dz[1:ie, 1:je] = self.pml.gi3[1:ie, :] * self.pml.gj3[:, 1:je] * self.Dz[1:ie, 1:je] + \
                              self.pml.gi2[1:ie, :] * self.pml.gj2[:, 1:je] * 0.5 * (self.Hy[1:ie, 1:je] - self.Hy[0:ie-1, 1:je] - \
                              self.Hx[1:ie, 1:je] + self.Hx[1:ie, 0:je-1])
    
    def update_Ez(self):
        # Calculate Ez
        self.Ez = self.gaz * (self.Dz - self.Iz)
        self.Iz = self.Iz + self.gbz * self.Ez

    def update_Hz(self):
        ie, je = self.nx, self.ny

        # Calculate Hz
        curl_e = (self.Ey[1:ie, :-1] - self.Ey[:-1, :-1]) - (self.Ex[:-1, 1:je] - self.Ex[:-1, :-1])
        self.iHz[:-1, :-1] = self.iHz[:-1, :-1] + curl_e
        self.Hz[:-1, :-1] = self.pml.fj3[:, :-1] * self.pml.fi3[:-1, :] * self.Hz[:-1, :-1] - \
                            self.pml.fj2[:, :-1] * self.pml.fi2[:-1, :] * (0.5 * curl_e + self.pml.fj1[:, :-1] * self.pml.fi1[:-1, :] * self.iHz[:-1, :-1])

    def update_Hx(self):
        ie, je = self.nx, self.ny

        # Calculate Hx
        curl_e = self.Ez[:-1, :-1] - self.Ez[:-1, 1:]
        self.iHx[:-1, :-1] = self.iHx[:-1, :-1] + curl_e
        self.Hx[:-1, :-1] = self.pml.fj3[:, :-1] * self.Hx[:-1, :-1] + \
                            self.pml.fj2[:, :-1] * (0.5 * curl_e + self.pml.fi1[:-1, :] * self.iHx[:-1, :-1])
    
    def update_Hy(self):
        ie, je = self.nx, self.ny

        # Calculate Hy
        curl_e = self.Ez[:-1, :-1] - self.Ez[1:, :-1]
        self.iHy[:-1, :-1] = self.iHy[:-1, :-1] + curl_e
        self.Hy[:-1, :-1] = self.pml.fi3[:-1, :] * self.Hy[:-1, :-1] - \
                            self.pml.fi2[:-1, :] * (0.5 * curl_e + self.pml.fj1[:, :-1] * self.iHy[:-1, :-1])

    # Update TE
    def update_fields_te(self):
        self.update_Dx()
        self.update_Ex()
        self.update_source(self.time, self.dt, self.Ex)
        self.update_Dy()
        self.update_Ey()
        self.update_Hz()

    # Update TM
    def update_fields_tm(self):
        self.update_Dz()
        self.update_Ez()
        self.update_source(self.time, self.dt, self.Ez)
        self.update_Hx()
        self.update_Hy()
    
    def simulation_step(self, time_step):
        self.actual_time_step = time_step
        self.update_fields()
        print(self.Ey[100,50])
        # Update time
        self.time += self.dt