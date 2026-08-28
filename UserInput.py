from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.data.StockCache import StockCache
from src.data.TimeSeries import TimeSeries


class UserInput:
    """Load, align, transform, and prepare user-supplied time series."""

    def __init__(
        self,
        stock_tickers: Sequence[str] | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        transform_type: str | None = None,
        time_series: TimeSeries | None = None,
        lag: int = 0,
        split: bool = True,
    ) -> None:
        self.stock_tickers = None if stock_tickers is None else list(stock_tickers)
        self.from_date = from_date
        self.to_date = to_date
        self.transform_type = transform_type
        self.lag = int(lag)
        self.split = split

        if self.lag < 0:
            raise ValueError("lag must be non-negative")

        raw_time_series = time_series or self._load_stock_time_series()
        raw_time_series.transform(transform_type=self.transform_type)
        self.time_series: list[TimeSeries] = raw_time_series.split(
            split=self.split
        )

    def _load_stock_time_series(self) -> TimeSeries:
        if not self.stock_tickers:
            raise ValueError("stock_tickers cannot be empty")

        if not self.split:
            if len(self.stock_tickers) != 1:
                raise ValueError("Forecasting requires exactly one stock ticker")
            ticker = self.stock_tickers[0]
            series = self._download_close(ticker)
            return TimeSeries(
                time=series.index.to_numpy(),
                rv_name=ticker,
                rv=series.to_numpy(),
            )

        if len(self.stock_tickers) != 2:
            raise ValueError("Conditional estimation requires exactly two stock tickers")

        prices = pd.concat(
            [self._download_close(ticker) for ticker in self.stock_tickers],
            axis=1,
            join="inner",
            keys=self.stock_tickers,
        )
        if self.lag:
            prices.iloc[:, 0] = prices.iloc[:, 0].shift(self.lag)
            prices = prices.dropna()

        return TimeSeries(
            time=prices.index.to_numpy(),
            covariate_name=self.stock_tickers[0],
            covariate=prices.iloc[:, 0].to_numpy(),
            rv_name=self.stock_tickers[1],
            rv=prices.iloc[:, 1].to_numpy(),
        )

    def _download_close(self, ticker: str) -> pd.Series:
        prices = StockCache().get_prices(
            ticker,
            start=self.from_date,
            end=self.to_date,
            interval="1d",
            auto_adjust=True,
            force_refresh=False,
            incremental=True,
        )

        close_data: pd.Series | pd.DataFrame = prices["Close"]

        if isinstance(close_data, pd.DataFrame):
            if close_data.shape[1] != 1:
                raise ValueError(
                    f"Expected one Close series for {ticker}, "
                    f"received {close_data.shape[1]}"
                )

            close_series = pd.Series(
                close_data.to_numpy(copy=False).reshape(-1),
                index=close_data.index,
                name=ticker,
            )
        else:
            close_series = close_data.rename(ticker)

        if close_series.empty:
            raise ValueError(
                f"No closing prices available for {ticker}"
            )

        return close_series
