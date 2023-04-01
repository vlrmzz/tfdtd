import torch

def plane_wave(self):
    self.Ez_inc = torch.zeros((1,self.grid_size_y), dtype=torch.float64)
    self.Hx_inc = torch.zeros((1,self.grid_size_y), dtype=torch.float64)
    return

def point_source():
    return