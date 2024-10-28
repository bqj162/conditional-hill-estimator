from StockTicker import StockTicker
from TimeSeries import TimeSeries

class UserInput:
    def __init__(self, stock_tickers=None, transform_type=None, time_series=None):
        self.stock_tickers = stock_tickers
        self.time_series = time_series
        self.transform_type = transform_type
        self.init()

    def init(self):
        if self.time_series is None:
            self.generate_time_series_from_stock_tickers()
        self.time_series.transform(transform_type = self.transform_type)
        self.time_series = self.time_series.split()


    def generate_time_series_from_stock_tickers(self):
        prices = []
        for stock_ticker in self.stock_tickers:
            price = StockTicker(stock_ticker).get_prices()
            prices.append(price.Open)
        prices = prices[0].join(prices[1])
        self.time_series = TimeSeries(time=prices.index.values, covariate=prices.values[:,0], rv=prices.values[:,1])
