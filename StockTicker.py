import yfinance as yf

class StockTicker:
    def __init__(self, ticker, start=None, end=None):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.prices = None

    def get_prices(self):
        if self.prices is None:
            self.prices = yf.download(self.ticker, start=self.start, end=self.end, auto_adjust=True)
        return self.prices