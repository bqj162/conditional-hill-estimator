import pandas as pd

from src.data.parser import parse_command_line_arguments
from .backtesting import run_single_backtest

def run_batch_backtests(
    args_list: list[list[str]],
    q_s: list[float],
    fitting_window: int,
) -> pd.DataFrame:
    results: list[pd.DataFrame] = []

    for args in args_list:
        user_input = parse_command_line_arguments(args, split=False)
        prepared_series = user_input.time_series

        if len(prepared_series) != 1:
            raise ValueError(
                "Batch forecasting requires exactly one prepared time series"
            )

        _, backtest_result = run_single_backtest(
            time_series=prepared_series[0],
            quantiles=q_s,
            fitting_window=fitting_window,
        )
        results.append(backtest_result)

    return pd.concat(results, ignore_index=True)
