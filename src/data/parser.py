from __future__ import annotations

import argparse
import shlex
from collections.abc import Sequence

import pandas as pd

from UserInput import UserInput
from .TimeSeries import TimeSeries


VALID_TRANSFORM_TYPES = (None, "log_diff")


def parse_command_line_arguments(
    argv: Sequence[str] | str | None = None,
    split: bool = True,
) -> UserInput:
    parser = argparse.ArgumentParser(
        description="Estimate conditional heavy-tail behaviour from market or CSV data."
    )
    parser.add_argument(
        "-s",
        "--stocks",
        dest="stock_tickers",
        help="Comma-separated tickers (two for conditional estimation; one for forecasting).",
    )
    parser.add_argument(
        "-t",
        "--transform_type",
        dest="transform_type",
        choices=["log_diff"],
    )
    parser.add_argument("-f", "--file_path", dest="time_series")
    parser.add_argument("-fd", "--from_date", dest="from_date")
    parser.add_argument("-td", "--to_date", dest="to_date")
    parser.add_argument("-l", "--lag", dest="lag", type=int, default=0)

    parsed_argv = shlex.split(argv) if isinstance(argv, str) else argv
    parsed = parser.parse_args(parsed_argv)

    if parsed.stock_tickers is not None and parsed.time_series is not None:
        parser.error("provide stock tickers or a CSV time series, not both")

    stock_tickers: list[str] | None = None
    time_series: TimeSeries | None = None
    if parsed.stock_tickers is not None:
        stock_tickers = [ticker.strip() for ticker in parsed.stock_tickers.split(",")]
        if any(not ticker for ticker in stock_tickers):
            parser.error("stock tickers cannot be empty")

        # Older examples repeated a ticker for univariate forecasting. Accept
        # that form while normalising the internal representation.
        if not split and len(stock_tickers) == 2 and stock_tickers[0] == stock_tickers[1]:
            stock_tickers = stock_tickers[:1]
    elif parsed.time_series is not None:
        time_series = parse_time_series_file(parsed.time_series)
    else:
        parser.error("provide either --stocks or --file_path")

    return UserInput(
        stock_tickers=stock_tickers,
        from_date=parsed.from_date,
        to_date=parsed.to_date,
        transform_type=parsed.transform_type,
        time_series=time_series,
        lag=parsed.lag,
        split=split,
    )


def parse_transform_type(transform_type: str | None) -> str | None:
    if transform_type not in VALID_TRANSFORM_TYPES:
        raise ValueError(
            f"Unknown transform type {transform_type!r}; expected one of {VALID_TRANSFORM_TYPES}"
        )
    return transform_type


def parse_time_series_file(filename: str) -> TimeSeries:
    frame = pd.read_csv(filename)
    if frame.shape[1] != 3:
        raise ValueError(
            "CSV input must contain exactly three columns: time, covariate, response"
        )

    return TimeSeries(
        time=frame.iloc[:, 0].to_numpy(),
        covariate_name=str(frame.columns[1]),
        covariate=frame.iloc[:, 1].to_numpy(),
        rv_name=str(frame.columns[2]),
        rv=frame.iloc[:, 2].to_numpy(),
    )
