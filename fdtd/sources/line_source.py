from .base_source import BaseSource
import numpy as np

class LineSource(BaseSource):
    def __init__(self, x, y1, y2, function, frequency):
        super().__init__(source_type="line", function=function)
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency

    def update_source(self, time, dt, ez):
        if self.function == 'sinusoidal':
            ez[self.x, self.y1:self.y2] = np.sin(self.omega * time)
        else:
            raise ValueError(f"Unsupported function type: {self.function}")
