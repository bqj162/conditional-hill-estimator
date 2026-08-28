from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from numpy.typing import NDArray

from src.data.TimeSeries import TimeSeries


class Forecast(NamedTuple):
    residual_ts: TimeSeries
    forecast_date: pd.Timestamp
    forecast_mu: float
    forecast_sigma: float


@dataclass
class AR1GARCH11:
    """AR(1)-GARCH(1,1) filter used before extreme-quantile estimation."""

    phi: float
    alpha_0: float
    alpha_1: float
    beta: float
    fitted_model: ARCHModelResult
    returns: pd.Series
    series_name: str = "resid"

    @classmethod
    def fit(
        cls,
        returns: pd.Series,
        name: str = "resid",
        start_values: NDArray[np.float64] | None = None,
    ) -> AR1GARCH11:
        if len(returns) < 3:
            raise ValueError("AR(1)-GARCH(1,1) fitting requires at least three observations")

        x_t = returns.iloc[1:].to_numpy(dtype=float)
        x_tm1 = returns.shift(1).dropna().to_numpy(dtype=float)
        denominator = float(np.dot(x_tm1, x_tm1))
        if denominator == 0:
            raise ValueError("Cannot fit AR(1) coefficient to a constant-zero lag series")

        phi = float(np.dot(x_t, x_tm1) / denominator)
        eps = returns.iloc[1:] - phi * returns.shift(1).dropna()
        eps.index = returns.index[1:]

        model = arch_model(
            eps,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            rescale=False,
        )
        fit_kwargs = {
            "disp": "off",
            "update_freq": 0,
            "options": {"maxiter": 100},
        }
        if start_values is None:
            fitted = model.fit(**fit_kwargs)
        else:
            fitted = model.fit(starting_values=start_values, **fit_kwargs)

        params = fitted.params
        return cls(
            phi=phi,
            alpha_0=_required_parameter(params, "omega"),
            alpha_1=_required_parameter(params, "alpha[1]"),
            beta=_required_parameter(params, "beta[1]"),
            fitted_model=fitted,
            returns=returns,
            series_name=name,
        )

    def forecast_1(self) -> Forecast:
        forecast = self.fitted_model.forecast(horizon=1, reindex=False)
        # variance = float(forecast.variance.iloc[-1, 0])
        variance_array = np.asarray(forecast.variance, dtype=float)

        if (
            variance_array.ndim != 2
            or variance_array.shape[0] == 0
            or variance_array.shape[1] == 0
        ):
            raise ValueError(
                f"Unexpected forecast variance shape: {variance_array.shape}"
            )

        variance = float(variance_array[-1, 0])

        if not np.isfinite(variance) or variance <= 0:
            raise FloatingPointError(
                f"GARCH forecast returned invalid variance: {variance}"
            )

        if variance < 0:
            raise ValueError("GARCH forecast returned a negative variance")

        forecast_mu = self.phi * float(self.returns.iloc[-1])
        residual_ts = _build_lagged_residual_time_series(
            raw_residuals=self.fitted_model.std_resid,
            expected_index=cast(pd.Index, self.returns.index[1:]),
            series_name=self.series_name,
        )

        last_date = pd.Timestamp(str(self.returns.index[-1]))
        next_business_day = cast(
            pd.Timestamp,
            last_date + pd.tseries.offsets.BDay(1),
        )
        return Forecast(
            residual_ts=residual_ts,
            forecast_date=next_business_day,
            forecast_mu=forecast_mu,
            forecast_sigma=float(np.sqrt(variance)),
        )


def _required_parameter(params: pd.Series, name: str) -> float:
    if name not in params.index:
        raise KeyError(f"Fitted GARCH model did not return parameter {name!r}")
    return float(params.loc[name])

def _build_lagged_residual_time_series(
    raw_residuals: pd.Series | NDArray[np.float64],
    expected_index: pd.Index,
    series_name: str,
) -> TimeSeries:
    """Create correctly dated consecutive residual pairs."""

    if isinstance(raw_residuals, pd.Series):
        if not raw_residuals.index.is_unique:
            raise ValueError("Residual dates must be unique")

        residuals = raw_residuals

        if not residuals.index.equals(expected_index):
            unexpected_dates = residuals.index.difference(expected_index)
            if not unexpected_dates.empty:
                raise ValueError(
                    "Fitted residual dates do not match the fitted return series"
                )

            # Only used in the unusual case where expected dates are missing.
            residuals = residuals.reindex(expected_index)

        values = residuals.to_numpy(dtype=float, copy=False)
        dates = residuals.index.to_numpy()
    else:
        values = np.asarray(raw_residuals, dtype=float)

        if values.ndim != 1:
            raise ValueError("Standardised residuals must be one-dimensional")
        if values.size != len(expected_index):
            raise ValueError(
                "Cannot reliably assign dates to standardised residuals"
            )

        dates = expected_index.to_numpy()

    previous = values[:-1]
    current = values[1:]

    # A pair is valid only when both Z_{t-1} and Z_t are finite.
    valid_pairs = np.isfinite(previous) & np.isfinite(current)

    if not np.any(valid_pairs):
        raise ValueError("No finite consecutive residual pairs are available")

    return TimeSeries(
        time=dates[1:][valid_pairs],
        covariate_name=f"{series_name}_lag1",
        covariate=previous[valid_pairs],
        rv_name=series_name,
        rv=current[valid_pairs],
    )
