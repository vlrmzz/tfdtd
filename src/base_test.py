import numpy as np
import torch
import pytorch_lightning as pl

import src.sources as sources

class FDTD2D(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        self.params = params
        # Initialize time
        self.time = 0.0
        self.setup_simulation()


    def setup_simulation(self):
        # Initialize simulation parameters, grid, and fields
        self.init_params()
        self.init_grid()
        self.init_pml()
        #self.init_plane_wave()
        # Set up PML, TFSF, and material distribution
        #self.setup_pml()
        #self.setup_tfsf()
        #self.setup_material_distribution()

    def init_params(self):
        # Simulation parameters
        self.nx = self.params.get('nx', 60)
        self.ny = self.params.get('ny', 60)
        self.time_steps = self.params.get('time_steps', 50)
        self.time = 0.0
        self.dx = self.params.get('dx', 1e-9)
        self.dy = self.params.get('dy', 1e-9)
        self.dt = self.dx / 6e8
        # PML
        self.use_pml = self.params.get('use_pml', True)
        self.pml_thickness = self.params.get('pml_thickness', 20)
        # TFSF
        self.use_tfsf = self.params.get('use_tfsf', False)
        self.tfsf_thickness = self.params.get('tfsf_thickness', 10)
        self.polarization = self.params.get('polarization', 'TM')
        # Source
        self.source_type = self.params.get('source_type', 'point_source')
         # Physical constants
        self.c = 299792458  # Speed of light in vacuum
        self.epsilon_0 = 8.85418782e-12  # Permittivity of free space
        self.mu_0 = 1.25663706e-6  # Permeability of free space


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
        self.ib = self.nx - self.ia - 1 
        self.jb = self.ny - self.ja - 1

        # Absorbing Boundary Conditions 
        self.boundary_low = [0, 0] 
        self.boundary_high = [0, 0]

    def simulation_step(self, time_step):
        self.actual_time_step = time_step
        ie, je = self.nx, self.ny

        # self.Ez_inc[:, 1:je] = self.Ez_inc[:, 1:je] + \
        #                     0.5 * (self.Hx_inc[:, 0:je-1] - self.Hx_inc[:, 1:je])

        # # Absorbing boundary conditions
        # self.Ez_inc[:, 0] = self.boundary_low.pop(0)
        # self.boundary_low.append(self.Ez_inc[:, 1])

        # self.Ez_inc[:, je-1] = self.boundary_high.pop(0)
        # self.boundary_high.append(self.Ez_inc[:, je-2])

        # Calculate Dz
        self.Dz[1:ie, 1:je] = self.gi3[1:ie, :] * self.gj3[:, 1:je] * self.Dz[1:ie, 1:je] + \
                                 self.gi2[1:ie, :] * self.gj2[:, 1:je] * 0.5 * \
                                 (self.Hy[1:ie, 1:je] - self.Hy[0:ie-1, 1:je] - self.Hx[1:ie, 1:je] + self.Hx[1:ie, 0:je-1])

        # Incident Dz values
        # self.Dz[self.ia:self.ib+1, self.ja] = self.Dz[self.ia:self.ib+1, self.ja] + \
        #                                     0.5 * self.Hx_inc[:, self.ja-1]
        # self.Dz[self.ia:self.ib+1, self.jb] = self.Dz[self.ia:self.ib+1, self.jb] - \
        #                                     0.5 * self.Hx_inc[:, self.jb-1]

        # Calculate Ez
        self.Ez = self.gaz * (self.Dz - self.Iz)
        self.Iz = self.Iz + self.gbz * self.Ez

        # # Incident Hx values
        # self.Hx_inc[:, 0:je-1] = self.Hx_inc[:, 0:je-1] + \
        #                     0.5 * (self.Ez_inc[:, 0:je-1] - self.Ez_inc[:, 1:je])

        # Calculate Hx
        # Calculate the curl of E-field
        curl_e = self.Ez[:-1, :-1] - self.Ez[:-1, 1:]
        # Update H-field in PML region
        self.iHx[:-1, :-1] = self.iHx[:-1, :-1] + curl_e
        self.Hx[:-1, :-1] = self.fj3[:, :-1] * self.Hx[:-1, :-1] + \
                            self.fj2[:, :-1] * (0.5 * curl_e + self.fi1[:-1, :] * self.iHx[:-1, :-1])

        # # Incident Hx values
        # self.Hx[:, self.ja-1] = self.Hx[:, self.ja-1] + \
        #                 0.5 * self.Ez_inc[:, self.ja]
        # self.Hx[:, self.jb] = self.Hx[:, self.jb] - \
        #               0.5 * self.Ez_inc[:, self.jb]

        # Calculate Hy
        # Calculate the curl of E-field
        curl_e = self.Ez[:-1, :-1] - self.Ez[1:, :-1]

        # Update H-field in PML region
        self.iHy[:-1, :-1] = self.iHy[:-1, :-1] + curl_e
        self.Hy[:-1, :-1] = self.fi3[:-1, :] * self.Hy[:-1, :-1] - \
                            self.fi2[:-1, :] * (0.5 * curl_e + self.fj1[:, :-1] * self.iHy[:-1, :-1])

        # # Incident Hy values
        # self.Hy[self.ia-1, self.ja:self.jb] = self.Hy[self.ia-1, self.ja:self.jb] - \
        #                                0.5 * self.Ez_inc[:, self.ja:self.jb]
        # self.Hy[self.ib-1, self.ja:self.jb] = self.Hy[self.ib-1, self.ja:self.jb] + \
        #                                0.5 * self.Ez_inc[:, self.ja:self.jb]

        # Add point source
        # pulse = np.exp(-0.5 * ((20 - time_step) / 8) ** 2)
        # self.Dz[100, 100] = pulse
            # Import and call the source function
        source = getattr(sources, self.source_type)
        source(self)


        # Update time
        self.time += self.dt