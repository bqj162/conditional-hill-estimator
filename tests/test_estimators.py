import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.TimeSeries import TimeSeries
from src.estimators.HillEstimator import HillEstimator
from src.estimators.quantile_estimator import QuantileSeries
from src.estimators.ts_fitting import _build_lagged_residual_time_series


class HillEstimatorTests(unittest.TestCase):
    def test_k_grid_uses_fractions_of_sample_size(self) -> None:
        estimator = HillEstimator(time_series=None, grid_resolution=5)

        grid = estimator._k_grid(upper_k_frac=0.5, n=500)

        np.testing.assert_array_equal(grid, np.array([5, 66, 127, 188, 250]))

    def test_k_grid_rejects_sample_too_small_for_minimum_k(self) -> None:
        estimator = HillEstimator(time_series=None)

        with self.assertRaisesRegex(ValueError, "Every k must be at least 2"):
            estimator._k_grid(upper_k_frac=0.5, n=100)

    def test_vectorized_estimator_matches_scalar_implementation(self) -> None:
        sample = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0])
        k_values = np.array([1, 2, 4])

        vectorized = HillEstimator.unconditional_hill_vectorized(sample, k_values)
        scalar = np.array(
            [
                HillEstimator.unconditional_hill_estimator(sample, int(k))
                for k in k_values
            ]
        )

        np.testing.assert_allclose(vectorized, scalar)

    def test_conditional_vectorized_matches_scalar_implementation(self) -> None:
        covariate = np.array([0.2, -0.1, 0.4, 0.8, 0.3, 0.6, -0.2, 0.9])
        response = np.array([1.0, 1.2, 1.5, 2.0, 2.7, 3.5, 5.0, 8.0])
        k_values = np.array([2, 3, 4])
        evaluation_point = 0.35

        vectorized = HillEstimator.gamma_fixed_k_n_x_vectorized(
            X=covariate,
            Y=response,
            k_array=k_values,
            x=evaluation_point,
        )
        scalar = [
            HillEstimator.gamma_fixed_k_n_x(
                X=covariate,
                Y=response,
                k_n=int(k),
                x=evaluation_point,
            )
            for k in k_values
        ]

        np.testing.assert_allclose(
            vectorized["gamma"].to_numpy(),
            np.array([result[0] for result in scalar]),
        )
        np.testing.assert_allclose(
            vectorized["q_n"].to_numpy(),
            np.array([result[1] for result in scalar]),
        )


class RollingForecastTests(unittest.TestCase):
    def test_forecast_dates_and_observations_are_one_step_ahead(self) -> None:
        dates = pd.date_range("2025-01-01", periods=6, freq="B").to_numpy()
        losses = np.arange(1.0, 7.0)
        series = TimeSeries(time=dates, rv_name="loss", rv=losses)
        estimator = QuantileSeries(series, q=[0.95], fitting_window=3)
        window_end_values: list[float] = []

        def fake_quantile_at_t(
            window: TimeSeries,
            start_values: np.ndarray | None,
        ) -> tuple[pd.DataFrame, np.ndarray]:
            window_end_values.append(float(window.rv[-1]))
            return (
                pd.DataFrame(
                    {
                        "date": [pd.Timestamp("1900-01-01")],
                        "q": [0.95],
                        "x_hat": [1.0],
                        "x_hat_unc": [1.0],
                        "obs": [np.nan],
                    }
                ),
                np.array([0.1, 0.2, 0.7]),
            )

        with patch.object(estimator, "quantile_at_t", side_effect=fake_quantile_at_t):
            result = estimator.quantile_all_dates(series, fitting_window=3)

        np.testing.assert_array_equal(result.index.to_numpy(), dates[3:])
        np.testing.assert_array_equal(result["obs"].to_numpy(), losses[3:])
        self.assertEqual(window_end_values, [3.0, 4.0, 5.0])


class ResidualAlignmentTests(unittest.TestCase):
    def test_internal_missing_residual_does_not_create_false_lag_pair(self) -> None:
        dates = pd.date_range("2025-01-01", periods=5, freq="B")
        residuals = pd.Series(
            [0.1, 0.2, np.nan, 0.4, 0.5],
            index=dates,
            dtype=float,
        )

        result = _build_lagged_residual_time_series(
            raw_residuals=residuals,
            expected_index=dates,
            series_name="resid",
        )

        assert result.time is not None
        assert result.covariate is not None
        np.testing.assert_array_equal(
            result.time,
            dates[[1, 4]].to_numpy(),
        )
        np.testing.assert_allclose(result.covariate, np.array([0.1, 0.4]))
        np.testing.assert_allclose(result.rv, np.array([0.2, 0.5]))


if __name__ == "__main__":
    unittest.main()
