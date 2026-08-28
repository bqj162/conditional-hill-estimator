from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.data.TimeSeries import TimeSeries
from .HillEstimator import HillEstimator
from .ts_fitting import AR1GARCH11


class QuantileSeries:
    """Rolling one-step-ahead conditional and unconditional tail forecasts."""

    def __init__(
        self,
        time_series: TimeSeries,
        q: Sequence[float],
        fitting_window: int,
    ) -> None:
        if not q or any(level <= 0 or level >= 1 for level in q):
            raise ValueError("All quantile levels must lie strictly between zero and one")
        if fitting_window < 3:
            raise ValueError("fitting_window must be at least three")

        self.time_series = time_series
        self.q = list(q)
        self.fitting_window = fitting_window
        self.quantile_series: pd.DataFrame | None = None

    def estimate(self) -> pd.DataFrame:
        self.quantile_series = self.quantile_all_dates(
            time_series=self.time_series,
            fitting_window=self.fitting_window,
        )
        return self.quantile_series

    def quantile_all_dates(
        self,
        time_series: TimeSeries,
        fitting_window: int,
    ) -> pd.DataFrame:
        if time_series.time is None:
            raise ValueError("Rolling forecasts require a time index")

        n = len(time_series.rv)
        if n <= fitting_window:
            raise ValueError("fitting_window must be smaller than the time-series length")

        results: list[pd.DataFrame] = []
        start_values: NDArray[np.float64] | None = None
        for forecast_index in range(fitting_window, n):
            window = TimeSeries(
                time=time_series.time[forecast_index - fitting_window : forecast_index],
                rv=time_series.rv[forecast_index - fitting_window : forecast_index],
                rv_name=time_series.rv_name,
            )
            quantiles, start_values = self.quantile_at_t(
                window,
                start_values=start_values,
            )

            # Use the observed market date instead of assuming every next date is
            # a business day; this also keeps the forecast and realised loss aligned.
            quantiles["date"] = time_series.time[forecast_index]
            quantiles["obs"] = float(time_series.rv[forecast_index])
            results.append(quantiles)

        output = pd.concat(results, ignore_index=True)
        return output.set_index("date")

    def quantile_at_t(
        self,
        time_series: TimeSeries,
        start_values: NDArray[np.float64] | None,
    ) -> tuple[pd.DataFrame, NDArray[np.float64]]:
        model = AR1GARCH11.fit(
            returns=time_series.to_rv_series(),
            name=time_series.rv_name,
            start_values=start_values,
        )
        fitting = model.forecast_1()

        residuals = fitting.residual_ts
        positive = residuals.rv > 0
        positive_ts = TimeSeries(
            time=None if residuals.time is None else residuals.time[positive],
            covariate_name=residuals.covariate_name,
            covariate=(
                None
                if residuals.covariate is None
                else residuals.covariate[positive]
            ),
            rv_name=residuals.rv_name,
            rv=residuals.rv[positive],
        )

        n = len(positive_ts.rv)
        if n < 3 or positive_ts.covariate is None:
            raise ValueError("Too few positive standardised residuals for tail estimation")

        # k_n = min(int(np.floor(n ** (2 / 3))), n - 1)
        k_n = int(np.floor(n/10))
        z_t = float(positive_ts.rv[-1])
        gamma, q_n = HillEstimator.gamma_fixed_k_n_x(
            X=positive_ts.covariate,
            Y=positive_ts.rv,
            k_n=k_n,
            x=z_t,
        )
        gamma_unc = HillEstimator.unconditional_hill_estimator(
            Y=positive_ts.rv,
            k_n=k_n,
        )
        order_stat = float(np.partition(positive_ts.rv, -k_n - 1)[-k_n - 1])

        output = pd.DataFrame(
            {
                "date": [fitting.forecast_date for _ in self.q],
                "q": self.q,
                "x_hat": [
                    fitting.forecast_mu
                    + fitting.forecast_sigma
                    * q_n
                    * ((1 - level) / (k_n / n)) ** (-gamma)
                    for level in self.q
                ],
                "x_hat_unc": [
                    fitting.forecast_mu
                    + fitting.forecast_sigma
                    * order_stat
                    * ((1 - level) / (k_n / n)) ** (-gamma_unc)
                    for level in self.q
                ],
                "obs": np.nan,
            }
        )
        fitted_parameters = model.fitted_model.params.to_numpy(dtype=float)
        return output, fitted_parameters


# Backwards-compatible alias for existing notebooks and scripts.
quantileSeries = QuantileSeries
