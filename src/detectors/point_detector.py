import numpy as np
from .base_detector import Detector

# PointDetector class
class PointDetector(Detector):
    def __init__(self, name, position):
        super().__init__(name, position)
        self.recorded_values = []

    def record(self, field_values):
        value = field_values[self.position[0], self.position[1]]
        self.recorded_values.append(value)