class BaseSource:
    def __init__(self, source_params):
        self.source_type = source_params['source_type']
        self.function = source_params['function']

    def update_source(self, time_step):
        raise NotImplementedError("This method should be implemented in the derived classes.")

    def __str__(self):
        return f"source_type: {self.source_type}, function: {self.function}"