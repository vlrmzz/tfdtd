class Detector:
    def __init__(self, name, position):
        self.name = name
        self.position = position

    def record(self, field_values):
        raise NotImplementedError("This method should be implemented in a subclass.")
