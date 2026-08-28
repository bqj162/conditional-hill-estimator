from __future__ import annotations

import pandas as pd
import yfinance as yf


class StockTicker:
    def __init__(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        self.ticker = ticker
        self.start = start
        self.end = end
        self.prices: pd.DataFrame | None = None

    def get_prices(self) -> pd.DataFrame:
        if self.prices is not None:
            return self.prices

        downloaded_prices = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            auto_adjust=True,
        )
        if downloaded_prices is None:
            raise ValueError(f"No price data returned for {self.ticker}")

        prices = downloaded_prices
        price_index = pd.DatetimeIndex(prices.index)
        if price_index.tz is None:
            prices.index = price_index.tz_localize("US/Eastern")

        self.prices = prices
        return prices
