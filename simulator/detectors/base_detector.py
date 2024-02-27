class Detector:
    def __init__(self, detector_params):
        self.name = detector_params['name']
        self.type = detector_params['type']
        self.recorded_values = []

    def record(self, field_values):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def __str__(self):
        return f"Detector: {self.name}, Type: {self.type}"

