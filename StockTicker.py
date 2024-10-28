import yfinance as yf

class StockTicker:
    def __init__(self, ticker, start="2010-01-01", end="2020-12-31"):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.prices_open = None

    def get_prices_open(self):
        if self.prices_open is None:
            self.prices_open = yf.download(self.ticker, start=self.start, end=self.end, auto_adjust=True)
        return self.prices_open