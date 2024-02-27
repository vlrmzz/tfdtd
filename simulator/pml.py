import torch
import numpy as np

class PML:
    def __init__(self, nx, ny, pml_thickness, backend, precision):
        self.backend = backend
        self.precision = precision
        self.nx = nx
        self.ny = ny
        self.pml_thickness = pml_thickness

        self.init_pml()
    def init_pml_coefficient(self, shape):
        if self.backend not in ('numpy', 'pytorch'):
            raise ValueError("Invalid backend. Choose either 'numpy' or 'pytorch'.")
        if self.precision not in ('float32', 'float64'):
            raise ValueError("Invalid dtype. Choose either 'float32' or 'float64'.")

        if self.backend == 'numpy':
            return np.ones(shape, dtype=self.precision)
        elif self.backend == 'pytorch':
            return torch.ones(shape, dtype=getattr(torch, self.precision))

    def init_pml(self):
        i, j = self.nx, self.ny

        # Init coefficients for PML TM polarization
        self.gi2 = self.init_pml_coefficient((i, 1))
        self.gi3 = self.init_pml_coefficient((i, 1))
        self.fi1 = self.init_pml_coefficient((i, 1)) - 1.0
        self.fi2 = self.init_pml_coefficient((i, 1))
        self.fi3 = self.init_pml_coefficient((i, 1))

        self.gj2 = self.init_pml_coefficient((1, j))
        self.gj3 = self.init_pml_coefficient((1, j))
        self.fj1 = self.init_pml_coefficient((1, j)) - 1.0
        self.fj2 = self.init_pml_coefficient((1, j))
        self.fj3 = self.init_pml_coefficient((1, j))

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