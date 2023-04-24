import torch
from .base_geometry import Geometry

class Circle(Geometry):
    def __init__(self, center_x, center_y, radius, epsr, sigma):
        super().__init__(epsr, sigma)
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

    def generate_mask(self, nx, ny):
        i, j = torch.meshgrid(torch.arange(0, nx), torch.arange(0, ny))
        xdist = (self.center_x - i)
        ydist = (self.center_y - j)
        dist = torch.sqrt(xdist ** 2 + ydist ** 2)
        circle_mask = (dist <= self.radius).double()
        return circle_mask