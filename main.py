import sys
import time

from src.plotting.Plots import Plots
from src.data.parser import parse_command_line_arguments
from src.backtesting.backtesting import run_single_backtest


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    user_input = parse_command_line_arguments(args, split=False)

    if len(user_input.time_series) != 1:
        raise ValueError("Single backtesting requires exactly one time series")

    time_series = user_input.time_series[0]
    quantiles = [0.95]
    fitting_window = 500

    started = time.perf_counter()
    forecasts, results = run_single_backtest(
        time_series=time_series,
        quantiles=quantiles,
        fitting_window=fitting_window,
    )
    elapsed = time.perf_counter() - started

    print(f"Estimation took {elapsed:.3f} seconds")
    print(results)

    plots = Plots(
        time_series=time_series,
        hill_estimate=None,
        quantile_fit=forecasts,
        q=quantiles[0],
    )
    plots.plot_fit()
    plots.plot_fitted_violations()


if __name__ == "__main__":
    main()
