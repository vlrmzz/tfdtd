from .base_source import BaseSource
import numpy as np

class LineSource(BaseSource):
    def __init__(self, source_params):
        super().__init__(source_params)
        self.x = source_params['x_start']
        self.y1 = source_params['y_start']
        self.y2 = source_params['y_end']

        if self.function == 'gaussian':
            self.amplitude = source_params['amplitude']
            self.t0 = source_params.get('t0', 0)
            self.frequency_center = float(source_params['frequency_center'])
            self.frequency_width = float(source_params['frequency_width'])
            self.sigma_f = self.frequency_width / (2.0 * np.sqrt(2 * np.log(2)))
            self.sigma_t = 1 / (2 * np.pi * self.sigma_f)
            self.omega = 2.0 * np.pi * self.frequency_center
        elif self.function == 'sinusoidal':
            self.frequency = float(source_params['frequency'])
            self.omega = 2.0 * np.pi * self.frequency
        else:
            raise ValueError(f"Unsupported function type: {self.function}")

    def update_source(self, time, dt, ez):
        if self.function == 'gaussian':
            ez[self.x, self.y1:self.y2] = self.amplitude * np.exp(-((time - self.t0) ** 2) / (2 * self.sigma_t ** 2)) * \
                                            np.sin(self.omega * (time-self.t0))
        elif self.function == 'sinusoidal':
            ez[self.x, self.y1:self.y2] = np.sin(self.omega * time)
        else:
            raise ValueError(f"Unsupported function type: {self.function}")

    def __str__(self):
        base_str = super().__str__()
        if self.function == 'gaussian':
            return f"{base_str}, source_x: {self.source_x}, source_y: {self.source_y}, amplitude: {self.amplitude}, t0: {self.t0}, frequency center: {self.frequency_center}, frequency width: {self.frequency_width}"
        elif self.function == 'sinusoidal':
            return f"{base_str}, x: {self.x}, y1: {self.y1}, y2: {self.y2}, frequency: {self.frequency}, omega: {self.omega}"