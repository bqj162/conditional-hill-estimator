import unittest
from unittest.mock import patch

import pandas as pd

from src.data.StockTicker import StockTicker


class StockTickerTests(unittest.TestCase):
    def test_get_prices_rejects_missing_download_result(self) -> None:
        with patch("src.data.StockTicker.yf.download", return_value=None):
            ticker = StockTicker("^GDAXI")

            with self.assertRaisesRegex(
                ValueError,
                r"No price data returned for \^GDAXI",
            ):
                ticker.get_prices()

    def test_get_prices_localizes_and_caches_downloaded_prices(self) -> None:
        downloaded = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.date_range("2025-01-02", periods=2, freq="B"),
        )

        with patch(
            "src.data.StockTicker.yf.download",
            return_value=downloaded,
        ) as download:
            ticker = StockTicker(
                "^GDAXI",
                start="2025-01-01",
                end="2025-01-10",
            )
            first = ticker.get_prices()
            second = ticker.get_prices()

        self.assertIs(first, second)
        self.assertEqual(str(pd.DatetimeIndex(first.index).tz), "US/Eastern")
        download.assert_called_once_with(
            "^GDAXI",
            start="2025-01-01",
            end="2025-01-10",
            auto_adjust=True,
        )
