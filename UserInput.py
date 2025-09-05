from StockTicker import StockTicker
from StockCache import StockCache
from TimeSeries import TimeSeries
import pandas as pd

class UserInput:
    def __init__(self, stock_tickers=None, from_date=None, to_date=None, transform_type=None, time_series=None,  lag: int = 0, split=None):
        self.stock_tickers = stock_tickers
        self.from_date = from_date
        self.to_date = to_date
        self.time_series = time_series
        self.transform_type = transform_type
        self.lag = int(lag)
        self.split = split
        self.init()

    def init(self):
        if self.time_series is None:
            self.generate_time_series_from_stock_tickers(split = self.split)
        self.time_series.transform(transform_type = self.transform_type)
        self.time_series = self.time_series.split(split=self.split)

    def generate_time_series_from_stock_tickers(self, split):
        cache = StockCache() 
        if (len([self.stock_tickers]) == 1) and not split:
            # series = (StockTicker(self.stock_tickers,start=self.from_date,end=self.to_date).get_prices().Close)
            df = cache.get_prices(self.stock_tickers, start=self.from_date, end=self.to_date,
                      interval='1d', auto_adjust=True, force_refresh=False, incremental=True)
            series = df['Close']
            self.time_series = TimeSeries(time=series.index.values, 
                                          rv_name = self.stock_tickers, 
                                          rv=series.values)
        else :
            prices = []
            for stock_ticker in self.stock_tickers:
                # price = StockTicker(stock_ticker, start=self.from_date, end=self.to_date).get_prices()
                df = cache.get_prices(stock_ticker, start=self.from_date, end=self.to_date,
                      interval='1d', auto_adjust=True, force_refresh=False, incremental=True)
                series = df['Close']
                # prices.append(price.Close)
                prices.append(series)
            prices = pd.concat(prices, axis=1, join="inner", keys=self.stock_tickers) # prices[0].join(prices[1], how="inner")
            self.time_series = TimeSeries(time=prices.index.values, 
                                        covariate_name=self.stock_tickers[0],  
                                        covariate=prices.values[:,0], 
                                        rv_name = self.stock_tickers[1], 
                                        rv=prices.values[:,1])
        # prices = []
        # for stock_ticker in self.stock_tickers:
        #     price = StockTicker(stock_ticker, start=self.from_date, end=self.to_date).get_prices()
        #     prices.append(price.Close)
        # prices = prices[0].join(prices[1], how="inner")
        # if self.lag > 0:
        #     cov, rv = self.stock_tickers
        #     prices[cov] = prices[cov].shift(self.lag)
        #     prices = prices.iloc[self.lag:]
        # self.time_series = TimeSeries(time=prices.index.values, covariate_name=self.stock_tickers[0],  covariate=prices.values[:,0], rv_name = self.stock_tickers[1], rv=prices.values[:,1])
