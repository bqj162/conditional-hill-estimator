class UserInput:
    def __init__(self, stock_tickers=None, transform_type=None, time_series=None):
        self.stock_tickers = stock_tickers
        self.time_series = time_series
        self.transform_type = transform_type
        self.time_series_transformed = None
        self.time_series_transformed_positive = None
        self.time_series_transformed_negative = None

    def get_time(self):
        return self.time_series['date']

    def get_covariate_process(self):
        return self.time_series['covariate']

    def get_rv_process(self):
        return self.time_series['rv']

    def transform_time_series(self):
        # implement
        # self.time_series_transformed = ...
        pass

    def split_time_series(self):
        # self.time_series_transformed_positive = ..
        # self.time_series_transformed_negative =
        pass


