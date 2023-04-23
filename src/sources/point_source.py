from .base_source import BaseSource
import numpy as np

class PointSource(BaseSource):
    def __init__(self, source_x, source_y, function, frequency):
        super().__init__(source_type="point", function=function)
        self.source_x = source_x
        self.source_y = source_y
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency

    def update_source(self, time, dt, ez):
        if self.function == 'sinusoidal':
            ez[self.source_x, self.source_y] = np.sin(self.omega * time)
        else:
            raise ValueError(f"Unsupported function type: {self.function}")
