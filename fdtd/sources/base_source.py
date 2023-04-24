class BaseSource:
    def __init__(self, source_type, function):
        self.source_type = source_type
        self.function = function

    def update_source(self, time_step):
        raise NotImplementedError("This method should be implemented in the derived classes.")
