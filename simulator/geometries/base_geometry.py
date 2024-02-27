class Geometry:
    def __init__(self, epsr, sigma):
        self.epsr = epsr
        self.sigma = sigma

    def generate_mask(self, nx, ny):
        raise NotImplementedError("This method should be implemented in a subclass.")
