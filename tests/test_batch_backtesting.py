import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.backtesting.batch_backtesting import run_batch_backtests
from src.data.TimeSeries import TimeSeries


class BatchBacktestingTests(unittest.TestCase):
    def test_single_prepared_series_is_passed_to_quantile_estimator(self) -> None:
        series = TimeSeries(rv_name="^GDAXI_loss", rv=[0.01, 0.02, 0.03])
        fitted = pd.DataFrame({"q": [0.95], "obs": [0.02], "x_hat": [0.03]})
        backtest_result = pd.DataFrame({"name": ["^GDAXI_loss"]})

        with (
            patch(
                "src.backtesting.batch_backtesting.parse_command_line_arguments",
                return_value=SimpleNamespace(time_series=[series]),
            ),
            patch(
                "src.backtesting.batch_backtesting.quantileSeries"
            ) as estimator,
            patch(
                "src.backtesting.batch_backtesting.back_test",
                return_value=backtest_result,
            ) as back_test,
        ):
            estimator.return_value.estimate.return_value = fitted
            result = run_batch_backtests(
                [["--stocks", "^GDAXI"]],
                q_s=[0.95],
                fitting_window=3,
            )

        estimator.assert_called_once_with(
            q=[0.95],
            time_series=series,
            fitting_window=3,
        )
        back_test.assert_called_once_with(fitted, [0.95], "^GDAXI_loss")
        pd.testing.assert_frame_equal(result, backtest_result)


if __name__ == "__main__":
    unittest.main()
