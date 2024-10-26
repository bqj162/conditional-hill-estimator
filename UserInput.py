class UserInput:
    def __init__(self, time_series, kernel):
        self.time_series = time_series
        self.kernel = kernel

    def get_time(self):
        return self.time_series['date']

    def get_covariate_process(self):
        return self.time_series['covariate']

    def get_rv_process(self):
        return self.time_series['rv']