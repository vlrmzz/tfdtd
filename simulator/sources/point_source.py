from .base_source import BaseSource
import numpy as np

class PointSource(BaseSource):
    def __init__(self, source_params):
        super().__init__(source_params)
        self.source_x = source_params['source_x']
        self.source_y = source_params['source_y']

        if self.function == 'gaussian_pulse':
            self.amplitude = source_params['amplitude']
            self.t0 = source_params['t0']
            self.frequency_center = source_params['frequency_center']
            self.frequency_width = source_params['frequency_width']
            self.sigma_f = self.frequency_width / (2 * np.sqrt(2 * np.log(2)))
            self.sigma_t = 1 / (2 * np.pi * self.sigma_f)
            self.omega = 2 * np.pi * self.frequency_center 

        elif self.function == 'sinusoidal':
            self.frequency = source_params['frequency']
            self.omega = 2 * np.pi * self.frequency
        else:
            raise ValueError(f"Unsupported function type: {self.function}")

    def update_source(self, time, dt, ez):
        if self.function == 'gaussian_pulse':
            ez[self.source_x, self.source_y] = self.amplitude * np.exp(-((time - self.t0) ** 2) / (2 * self.sigma_t ** 2)) * \
                                            np.sin(self.omega * (time-self.t0))
        elif self.function == 'sinusoidal':
            ez[self.source_x, self.source_y] = np.sin(self.omega * time)
        else:
            raise ValueError(f"Unsupported function type: {self.function}")

    def __str__(self):
        base_str = super().__str__()
        if self.function == 'gaussian_pulse':
            return f"{base_str}, source_x: {self.source_x}, source_y: {self.source_y}, amplitude: {self.amplitude}, t0: {self.t0}, frequency center: {self.frequency_center}, frequency width: {self.frequency_width}"
        elif self.function == 'sinusoidal':
            return f"{base_str}, source_x: {self.source_x}, source_y: {self.source_y}, frequency: {self.frequency}, omega: {self.omega}"