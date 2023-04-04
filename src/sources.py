# import torch
# import numpy as np

# def point_source(sim):

#     pulse = np.exp(-0.5 * ((20 - sim.actual_time_step) / 8) ** 2)
#     sim.Dz[100, 100] = pulse

# def plane_wave(sim):

#     sim.Ez_inc = torch.zeros((1,sim.ny), dtype=torch.float64)
#     sim.Hx_inc = torch.zeros((1,sim.ny), dtype=torch.float64)
#     sim.ib = sim.nx - sim.ia - 1 
#     sim.jb = sim.ny - sim.ja - 1
#     # Absorbing Boundary Conditions 
#     sim.boundary_low = [0, 0] 
#     sim.boundary_high = [0, 0]
    
#     pulse = np.exp(-0.5 * ((20 - sim.actual_time_step) / 8) ** 2)
#     sim.Dz[75:125, 100] = pulse