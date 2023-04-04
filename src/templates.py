simulation_step_template = '''
def simulation_step(self, time_step):
    self.actual_time_step = time_step
    ie, je = self.nx, self.ny

    {update_Ez_inc}

    {set_boundaries}

    # Calculate Dz
    self.Dz[1:ie, 1:je] = self.gi3[1:ie, :] * self.gj3[:, 1:je] * self.Dz[1:ie, 1:je] + \
                        self.gi2[1:ie, :] * self.gj2[:, 1:je] * 0.5 * \
                        (self.Hy[1:ie, 1:je] - self.Hy[0:ie-1, 1:je] - self.Hx[1:ie, 1:je] + self.Hx[1:ie, 0:je-1])

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

    {add_source}

    # Update time
    self.time += self.dt
'''

point_source ='''
    import numpy as np
    pulse = np.exp(-0.5 * ((20 - self.actual_time_step) / 8) ** 2)
    self.Dz[100, 100] = pulse
'''