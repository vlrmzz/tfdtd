import numpy as np
import torch
from .base_detector import Detector

class PointDetector(Detector):
    def __init__(self, detector_params):
        super().__init__(detector_params)
        self.x = detector_params['x']
        self.y = detector_params['y']

    def record(self, field_values):
        value = field_values[self.x, self.y]
        self.recorded_values.append(torch.tensor([value]))

    def __str__(self):
        return super().__str__() + f", Position: ({self.x},{self.y})"
