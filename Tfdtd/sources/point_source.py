from .base_source import BaseSource
import numpy as np

class PointSource(BaseSource):
    def __init__(self, source_params):
        super().__init__(source_params)
        self.source_x = source_params['source_x']
        self.source_y = source_params['source_y']
        self.frequency = source_params['frequency']
        self.omega = 2 * np.pi * self.frequency

    def update_source(self, time, dt, ez):
        if self.function == 'sinusoidal':
            ez[self.source_x, self.source_y] = np.sin(self.omega * time)
        else:
            raise ValueError(f"Unsupported function type: {self.function}")

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}, source_x: {self.source_x}, source_y: {self.source_y}, frequency: {self.frequency}, omega: {self.omega}"