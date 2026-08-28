import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import main as main_module
from src.data.TimeSeries import TimeSeries


class MainTests(unittest.TestCase):
    def test_main_runs_single_backtest_and_plots_all_forecasts(self) -> None:
        series = TimeSeries(rv_name="^GDAXI_loss", rv=[0.1, 0.2, 0.3])
        forecast_dates = pd.to_datetime(["2007-12-31", "2008-01-02", "2010-12-31", "2011-01-03"])
        forecasts = pd.DataFrame(
            {"q": [0.95] * 4, "x_hat": [1.0] * 4},
            index=forecast_dates,
        )
        results = pd.DataFrame({"name": ["^GDAXI_loss"]})

        with (
            patch.object(
                main_module,
                "parse_command_line_arguments",
                return_value=SimpleNamespace(time_series=[series]),
            ) as parse_arguments,
            patch.object(
                main_module,
                "run_single_backtest",
                return_value=(forecasts, results),
            ) as run_single_backtest,
            patch.object(main_module, "Plots") as plots,
            patch.object(
                main_module.time,
                "perf_counter",
                side_effect=[10.0, 12.0],
            ),
        ):
            main_module.main(["--stocks", "^GDAXI"])

        parse_arguments.assert_called_once_with(
            ["--stocks", "^GDAXI"],
            split=False,
        )
        run_single_backtest.assert_called_once_with(
            time_series=series,
            quantiles=[0.95],
            fitting_window=500,
        )
        self.assertIs(plots.call_args.kwargs["time_series"], series)
        self.assertEqual(plots.call_args.kwargs["q"], 0.95)
        self.assertIs(plots.call_args.kwargs["quantile_fit"], forecasts)
        plots.return_value.plot_fit.assert_called_once_with()
        plots.return_value.plot_fitted_violations.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
