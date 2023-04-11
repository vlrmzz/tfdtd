import numpy as np
simulation_step_template = '''
def simulation_step(self, time_step):
    self.actual_time_step = time_step
    ie, je = self.nx, self.ny

    {update_Ez_inc}

    {set_boundaries}

    # Calculate Dz
    self.Dz[1:ie, 1:je] = self.gi3[1:ie, :] * self.gj3[:, 1:je] * self.Dz[1:ie, 1:je] + \
                            self.gi2[1:ie, :] * self.gj2[:, 1:je] * 0.5 * (self.Hy[1:ie, 1:je] - self.Hy[0:ie-1, 1:je] - \
                            self.Hx[1:ie, 1:je] + self.Hx[1:ie, 0:je-1])

    {add_source}

    {update_Dz_inc}

    # Calculate Ez
    self.Ez = self.gaz * (self.Dz - self.Iz)
    self.Iz = self.Iz + self.gbz * self.Ez

    {update_Hx_inc}

    # Calculate Hx
    # Calculate the curl of E-field
    curl_e = self.Ez[:-1, :-1] - self.Ez[:-1, 1:]
    # Update H-field in PML region
    self.iHx[:-1, :-1] = self.iHx[:-1, :-1] + curl_e
    self.Hx[:-1, :-1] = self.fj3[:, :-1] * self.Hx[:-1, :-1] + \
                        self.fj2[:, :-1] * (0.5 * curl_e + self.fi1[:-1, :] * self.iHx[:-1, :-1])

    {update_Hy_inc}

    #Calculate Hy
    # Calculate the curl of E-field
    curl_e = self.Ez[:-1, :-1] - self.Ez[1:, :-1]

    # Update H-field in PML region
    self.iHy[:-1, :-1] = self.iHy[:-1, :-1] + curl_e
    self.Hy[:-1, :-1] = self.fi3[:-1, :] * self.Hy[:-1, :-1] - \
                        self.fi2[:-1, :] * (0.5 * curl_e + self.fj1[:, :-1] * self.iHy[:-1, :-1])

    

    # Update time
    self.time += self.dt
'''

# Point source
point_source ='''
    import numpy as np
    pulse = np.exp(-0.5 * ((20 - self.actual_time_step) / 8) ** 2)
    self.Dz[100, 100] = pulse
'''

# Plane wave 
update_Ez_inc = '''
    self.Ez_inc[:, 1:je] = self.Ez_inc[:, 1:je] + \
                        0.5 * (self.Hx_inc[:, 0:je-1] - self.Hx_inc[:, 1:je])
'''

set_boundaries = '''
    # Absorbing boundary conditions
    self.Ez_inc[:, 0] = self.boundary_low.pop(0)
    self.boundary_low.append(self.Ez_inc[:, 1])

    self.Ez_inc[:, je-1] = self.boundary_high.pop(0)
    self.boundary_high.append(self.Ez_inc[:, je-2])
'''

update_Dz_inc = '''
    # Incident Dz values
    self.Dz[self.ia:self.ib+1, self.ja] = self.Dz[self.ia:self.ib+1, self.ja] + \
                                        0.5 * self.Hx_inc[:, self.ja-1]
    self.Dz[self.ia:self.ib+1, self.jb] = self.Dz[self.ia:self.ib+1, self.jb] - \
                                        0.5 * self.Hx_inc[:, self.jb-1]
'''

update_Hx_inc = '''
    # Incident Hx values
    self.Hx_inc[:, 0:je-1] = self.Hx_inc[:, 0:je-1] + \
                        0.5 * (self.Ez_inc[:, 0:je-1] - self.Ez_inc[:, 1:je])
'''

update_Hy_inc = '''
    # Incident Hy values
    self.Hy[self.ia-1, self.ja:self.jb] = self.Hy[self.ia-1, self.ja:self.jb] - \
                                    0.5 * self.Ez_inc[:, self.ja:self.jb]
    self.Hy[self.ib-1, self.ja:self.jb] = self.Hy[self.ib-1, self.ja:self.jb] + \
                                    0.5 * self.Ez_inc[:, self.ja:self.jb]
'''

plane_source = '''
    import numpy as np
    #pulse = np.exp(-0.5 * ((20 - self.actual_time_step) / 8) ** 2)
    pulse = np.sin(2 * np.pi * self.frequency * self.time)
    self.Dz[self.ia:self.ib, self.ja] = pulse
'''