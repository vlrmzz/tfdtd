import torch
from .base_geometry import Geometry

class Rectangle(Geometry):
    def __init__(self, x1, y1, x2, y2, epsr, sigma):
        super().__init__(epsr, sigma)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def generate_mask(self, nx, ny):
        i, j = torch.meshgrid(torch.arange(0, nx), torch.arange(0, ny))
        rectangle_mask = ((i >= self.x1) & (i <= self.x2) & (j >= self.y1) & (j <= self.y2)).double()
        return rectangle_mask
        
