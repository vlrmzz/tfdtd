# Utilities for fdtd simulation

import torch

def circle_primitive(nx,ny,center_x,center_y,radius,epsr,epsz,sigma,gaz,gbz,dt):
    i, j = torch.meshgrid(torch.arange(0, nx), torch.arange(0, ny))
    xdist = (center_x - i)
    ydist = (center_y - j)
    dist = torch.sqrt(xdist ** 2 + ydist ** 2)
    radius_mask = (dist <= radius).double()
    gaz[0:nx, 0:ny] = radius_mask * (1 / (epsr + (sigma * dt / epsz))) \
                            + (1 - radius_mask) * gaz[0:nx, 0:ny]
    gbz[0:nx, 0:ny] = radius_mask * ((sigma * dt / epsz)) \

    return gaz, gbz

def rectangle_primitive(nx, ny, x_min, x_max, y_min, y_max, epsr, epsz, sigma, gaz, gbz, dt):
    i, j = torch.meshgrid(torch.arange(0, nx), torch.arange(0, ny))

    # Create a mask for the rectangle's area
    rectangle_mask = ((i >= x_min) & (i <= x_max) & (j >= y_min) & (j <= y_max)).double()

    gaz[0:nx, 0:ny] = rectangle_mask * (1 / (epsr + (sigma * dt / epsz))) \
                            + (1 - rectangle_mask) * gaz[0:nx, 0:ny]
    gbz[0:nx, 0:ny] = rectangle_mask * ((sigma * dt / epsz))

    return gaz, gbz