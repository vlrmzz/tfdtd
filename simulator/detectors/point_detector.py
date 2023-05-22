import numpy as np
import torch
from .base_detector import Detector

# PointDetector class
class PointDetector(Detector):
    def __init__(self, detector_params):
        super().__init__(detector_params)
        self.recorded_values = []

    def record(self, field_values):
        value = field_values[self.position[0], self.position[1]]
        self.recorded_values.append(torch.tensor([value]))

    def __str__(self):
        return f"Detector: {self.name}, Position: {self.position}"