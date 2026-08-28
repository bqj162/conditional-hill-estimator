import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.backtesting.batch_backtesting import run_batch_backtests
from src.data.TimeSeries import TimeSeries


class BatchBacktestingTests(unittest.TestCase):
    def test_batch_delegates_each_series_to_single_backtest(self) -> None:
        series = TimeSeries(rv_name="^GDAXI_loss", rv=[0.01, 0.02, 0.03])
        fitted = pd.DataFrame({"q": [0.95], "obs": [0.02], "x_hat": [0.03]})
        backtest_result = pd.DataFrame({"name": ["^GDAXI_loss"]})

        with (
            patch(
                "src.backtesting.batch_backtesting.parse_command_line_arguments",
                return_value=SimpleNamespace(time_series=[series]),
            ),
            patch(
                "src.backtesting.batch_backtesting.run_single_backtest",
                return_value=(fitted, backtest_result),
            ) as run_single_backtest,
        ):
            result = run_batch_backtests(
                [["--stocks", "^GDAXI"]],
                q_s=[0.95],
                fitting_window=3,
            )

        run_single_backtest.assert_called_once_with(
            time_series=series,
            quantiles=[0.95],
            fitting_window=3,
        )
        pd.testing.assert_frame_equal(result, backtest_result)


if __name__ == "__main__":
    unittest.main()
