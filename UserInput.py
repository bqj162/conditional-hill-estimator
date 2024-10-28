
class UserInput:
    def __init__(self, stock_tickers=None, transform_type=None, time_series=None):
        self.stock_tickers = stock_tickers
        self.time_series = time_series
        self.transform_type = transform_type
        self.init()

    def init(self):
        self.time_series.transform(transform_type = self.transform_type)
        self.time_series = self.time_series.split()
