import numpy as np
import torch
from .base_detector import Detector

class LineDetector(Detector):
    def __init__(self, detector_params):
        super().__init__(detector_params)
        self.x_start = detector_params['x_start']
        self.x_end = detector_params['x_end']
        self.y_start = detector_params['y_start']
        self.y_end = detector_params['y_end']

    def record(self, field_values): # this logic does make sense to be refactored
        for x in range(self.x_start, self.x_end + 1):
            for y in range(self.y_start, self.y_end + 1):
                value = field_values[x, y]
                self.recorded_values.append(torch.tensor([value]))

    def __str__(self):
        return super().__str__() + f", Position: ({self.x_start},{self.y_start}) to ({self.x_end},{self.y_end})"
