import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.StockCache import StockCache
from src.data.parser import parse_command_line_arguments


class UserInputTests(unittest.TestCase):
    def test_repeated_univariate_ticker_returns_one_loss_series(self) -> None:
        dates = pd.date_range("2025-01-02", periods=3, freq="B")
        columns = pd.MultiIndex.from_tuples(
            [("Close", "^GDAXI")],
            names=["Price", "Ticker"],
        )
        prices = pd.DataFrame(
            [[100.0], [110.0], [99.0]],
            index=dates,
            columns=columns,
        )

        with patch.object(StockCache, "get_prices", return_value=prices) as get_prices:
            user_input = parse_command_line_arguments(
                [
                    "--stocks",
                    "^GDAXI,^GDAXI",
                    "--from_date",
                    "2025-01-02",
                    "--to_date",
                    "2025-01-07",
                    "--transform_type",
                    "log_diff",
                ],
                split=False,
            )

        self.assertEqual(user_input.stock_tickers, ["^GDAXI"])
        self.assertEqual(len(user_input.time_series), 1)

        loss_series = user_input.time_series[0]
        self.assertEqual(loss_series.rv_name, "^GDAXI_loss")
        assert loss_series.time is not None
        np.testing.assert_allclose(
            loss_series.rv,
            -np.diff(np.log(np.array([100.0, 110.0, 99.0]))),
        )
        np.testing.assert_array_equal(loss_series.time, dates[1:].to_numpy())
        get_prices.assert_called_once_with(
            "^GDAXI",
            start="2025-01-02",
            end="2025-01-07",
            interval="1d",
            auto_adjust=True,
            force_refresh=False,
            incremental=True,
        )


if __name__ == "__main__":
    unittest.main()
