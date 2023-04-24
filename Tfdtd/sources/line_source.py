from .base_source import BaseSource
import numpy as np

class LineSource(BaseSource):
    def __init__(self, source_params):
        super().__init__(source_params)
        self.x = source_params['x']
        self.y1 = source_params['y1']
        self.y2 = source_params['y2']
        self.frequency = source_params['frequency']
        self.omega = 2 * np.pi * self.frequency

    def update_source(self, time, dt, ez):
        if self.function == 'sinusoidal':
            ez[self.x, self.y1:self.y2] = np.sin(self.omega * time)
        else:
            raise ValueError(f"Unsupported function type: {self.function}")

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}, x: {self.x}, y1: {self.y1}, y2: {self.y2}, frequency: {self.frequency}, omega: {self.omega}"