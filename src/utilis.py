# Utilities for fdtd simulation

import torch

def circle_primitive(size_x,size_y,center_x,center_y,radius,epsr,epsz,sigma,gaz,gbz,dt):
    i, j = torch.meshgrid(torch.arange(0, size_x), torch.arange(0, size_y))
    xdist = (center_x - i)
    ydist = (center_y - j)
    dist = torch.sqrt(xdist ** 2 + ydist ** 2)
    radius_mask = (dist <= radius).double()

    gaz[0:size_x, 0:size_y] = radius_mask * (1 / (epsr + (sigma * dt / epsz))) \
                            + (1 - radius_mask) * gaz[0:size_x, 0:size_y]
    gbz[0:size_x, 0:size_y] = radius_mask * ((sigma * dt / epsz)) \

    return gaz, gbz