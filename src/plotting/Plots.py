from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.TimeSeries import TimeSeries


class Plots:
    def __init__(
        self,
        time_series: TimeSeries | None = None,
        hill_estimate: Mapping[str, Any] | None = None,
        quantile_fit: pd.DataFrame | None = None,
        q: float | None = None,
    ) -> None:
        self.time_series = time_series
        self.hill_estimate = hill_estimate
        self.quantile_fit = quantile_fit
        self.q = q

    def plot_2d_marginal(self) -> go.Figure:
        series = self._require_time_series()
        if series.covariate is None or series.covariate_name is None:
            raise ValueError("A covariate is required for the marginal plot")

        figure = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                f"Covariate process: {series.covariate_name}",
                f"Regularly varying process: {series.rv_name}",
            ),
        )
        figure.add_trace(
            go.Scatter(x=series.time, y=series.covariate, showlegend=False),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(x=series.time, y=series.rv, showlegend=False),
            row=2,
            col=1,
        )
        return figure

    def plot_3d(self) -> go.Figure:
        series = self._require_time_series()
        estimate = self._require_hill_estimate()
        figure = go.Figure(
            data=[
                go.Surface(
                    z=estimate["gamma_k_x"],
                    y=estimate["X"],
                    x=estimate["K"],
                )
            ]
        )
        figure.update_layout(
            title=(
                f"Conditional tail-index estimates: {series.covariate_name} "
                f"and {series.rv_name}"
            ),
            autosize=True,
        )
        figure.update_scenes(
            xaxis_title_text="k_n",
            yaxis_title_text="x",
            zaxis_title_text="gamma_k_x",
        )
        return figure

    def plot_fit(self) -> None:
        series = self._require_time_series()
        fit = self._require_quantile_fit()
        q = self._require_quantile_level()
        _, axis = plt.subplots()
        axis.bar(fit.index, fit["obs"], width=1.0, label="Log returns")
        axis.plot(
            fit.index,
            fit["x_hat"],
            label="Conditional",
            linewidth=0.7,
            linestyle="dashed",
            color="red",
        )
        axis.plot(
            fit.index,
            fit["x_hat_unc"],
            label="Unconditional",
            linewidth=0.7,
            linestyle="dashdot",
            color="green",
        )
        self._format_forecast_plot(axis, series)
        date_range = self._date_range_component(fit.index)
        self._save_and_show(
            f"{self._safe_component(series.rv_name)}_{date_range}_q{q}_quantiles.pdf",
            output_dir=Path("Plots") / "forecasting",
            metadata={
                "Title": f"{series.rv_name} loss-quantile forecasts (q={q})",
                "Subject": f"Forecast dates: {self._date_range_description(fit.index)}",
            },
        )

    def plot_fitted_violations(self) -> None:
        series = self._require_time_series()
        fit = self._require_quantile_fit()
        q = self._require_quantile_level()
        _, axis = plt.subplots()
        axis.bar(fit.index, fit["obs"], width=1.0, label="Log returns")

        conditional_exceedance = fit["obs"] > fit["x_hat"]
        unconditional_exceedance = fit["obs"] > fit["x_hat_unc"]
        axis.scatter(
            fit.index[conditional_exceedance],
            fit.loc[conditional_exceedance, "x_hat"],
            marker="o",
            color="red",
            s=20,
            label="Conditional exceedance",
        )
        axis.scatter(
            fit.index[unconditional_exceedance],
            fit.loc[unconditional_exceedance, "x_hat_unc"],
            marker="^",
            color="green",
            s=20,
            label="Unconditional exceedance",
        )
        self._format_forecast_plot(axis, series)
        date_range = self._date_range_component(fit.index)
        self._save_and_show(
            f"{self._safe_component(series.rv_name)}_{date_range}_q{q}_violations.pdf",
            output_dir=Path("Plots") / "forecasting",
            metadata={
                "Title": f"{series.rv_name} loss-quantile violations (q={q})",
                "Subject": f"Forecast dates: {self._date_range_description(fit.index)}",
            },
        )

    @staticmethod
    def plot_bias_MSE(bias_MSE_df: pd.DataFrame) -> None:
        k = bias_MSE_df["k"]
        figure, (bias_axis, mse_axis) = plt.subplots(1, 2)
        bias_axis.plot(k, bias_MSE_df["bias"], label="Unconditional", color="red")
        bias_axis.plot(k, bias_MSE_df["bias_t"], label="Conditional", color="green")
        bias_axis.axhline(y=0, color="black", linestyle="--", linewidth=1)
        mse_axis.plot(k, bias_MSE_df["MSE"], label="Unconditional", color="red")
        mse_axis.plot(k, bias_MSE_df["MSE_t"], label="Conditional", color="green")
        bias_axis.set(xlabel="k_n", ylabel="Bias")
        mse_axis.set(xlabel="k_n", ylabel="MSE")
        bias_axis.legend()
        mse_axis.legend()
        figure.tight_layout()
        plt.show()

    @staticmethod
    def plot_gamma_sim(gamma_df: pd.DataFrame) -> None:
        figure, axis = plt.subplots()
        axis.plot(gamma_df["k"], gamma_df["gamma"], label="Unconditional", color="red")
        axis.plot(gamma_df["k"], gamma_df["gamma_x"], label="Conditional", color="green")
        axis.plot(
            gamma_df["k"],
            gamma_df["target"],
            color="black",
            linestyle="--",
            label="Target",
        )
        axis.set(xlabel="k_n", ylabel="Tail index")
        axis.legend()
        figure.tight_layout()
        plt.show()

    @staticmethod
    def plot_2x2_grid_param(
        results_df: pd.DataFrame,
        burn_ins: Sequence[int],
        col_param: str,
        col_values: Sequence[Any],
        family: str,
        figsize: tuple[float, float] = (12, 8),
        linewidth: float = 1,
    ) -> None:
        if len(burn_ins) < 2 or len(col_values) < 2:
            raise ValueError("Provide at least two burn-ins and two column values")
        family = family.strip()
        if not family:
            raise ValueError("The simulation family must be named")

        b0, b1 = int(burn_ins[0]), int(burn_ins[1])
        c0, c1 = col_values[0], col_values[1]
        results_df = results_df.sort_values([col_param, "burn_in", "k"])
        figure, axes = plt.subplots(2, 2, figsize=figsize, sharex="col")
        style_map = {
            ("uncond", b0): ("red", "-", f"Uncond, n={b0}"),
            ("uncond", b1): ("red", "--", f"Uncond, n={b1}"),
            ("cond", b0): ("green", "-", f"Cond, n={b0}"),
            ("cond", b1): ("green", "-.", f"Cond, n={b1}"),
        }
        parameter_label = "x" if col_param == "x_eval" else col_param

        def plot_subplot(axis: Any, column_value: Any, metric: str) -> None:
            for (estimator, burn_in), (color, style, label) in style_map.items():
                selected = results_df[
                    (results_df[col_param] == column_value)
                    & (results_df["burn_in"] == burn_in)
                ]
                if selected.empty:
                    continue
                k_over_n = np.asarray(selected["k"], dtype=float) / burn_in
                column = metric if estimator == "uncond" else f"{metric}_t"
                axis.plot(
                    k_over_n,
                    np.asarray(selected[column], dtype=float),
                    label=label,
                    color=color,
                    linestyle=style,
                    linewidth=linewidth,
                )

            if metric == "bias":
                axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
            metric_label = metric.upper() if metric == "MSE" else "Bias"
            axis.set(
                xlabel="k/n",
                ylabel=metric_label,
                title=f"{metric_label}, {parameter_label} = {column_value}",
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize="small")

        plot_subplot(axes[0, 0], c0, "bias")
        plot_subplot(axes[0, 1], c1, "bias")
        plot_subplot(axes[1, 0], c0, "MSE")
        plot_subplot(axes[1, 1], c1, "MSE")
        figure.suptitle(f"{family} innovations: simulation bias and MSE")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        values = "_".join(Plots._safe_component(value) for value in col_values)
        Plots._save_and_show(
            f"bias_mse_{Plots._safe_component(col_param)}_{values}.pdf",
            output_dir=(
                Path("Plots")
                / "simulation"
                / Plots._safe_component(family).lower()
            ),
            metadata={
                "Title": f"{family} simulation: bias and MSE by {col_param}",
                "Subject": (
                    f"Sample sizes: {b0}, {b1}; "
                    f"{col_param} values: {c0}, {c1}"
                ),
                "Keywords": "conditional Hill estimator, simulation, bias, MSE",
            },
        )

    def _require_time_series(self) -> TimeSeries:
        if self.time_series is None:
            raise ValueError("This plot requires a time series")
        return self.time_series

    def _require_hill_estimate(self) -> Mapping[str, Any]:
        if self.hill_estimate is None:
            raise ValueError("This plot requires a conditional Hill estimate")
        return self.hill_estimate

    def _require_quantile_fit(self) -> pd.DataFrame:
        if self.quantile_fit is None:
            raise ValueError("This plot requires fitted quantiles")
        if self.quantile_fit.empty:
            raise ValueError("The fitted-quantile data must not be empty")
        return self.quantile_fit

    def _require_quantile_level(self) -> float:
        if self.q is None:
            raise ValueError("This plot requires a quantile level")
        return self.q

    def _format_forecast_plot(self, axis: Any, series: TimeSeries) -> None:
        axis.set_xlabel("Date")
        axis.set_ylabel("Loss return")
        axis.set_title(f"{series.rv_name}, q = {self.q}")
        axis.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()

    @staticmethod
    def _date_range_component(index: pd.Index) -> str:
        return Plots._safe_component(Plots._date_range_description(index))

    @staticmethod
    def _date_range_description(index: pd.Index) -> str:
        values = np.asarray(index)
        if values.size == 0:
            raise ValueError("Cannot describe an empty date range")
        if np.issubdtype(values.dtype, np.datetime64):
            dates = values.astype("datetime64[D]").astype(str)
            return f"{dates[0]}_{dates[-1]}"
        return f"{values[0]}_{values[-1]}"

    @staticmethod
    def _safe_component(value: Any) -> str:
        component = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).lstrip("^"))
        return component.strip("-_") or "unnamed"

    @staticmethod
    def _save_and_show(
        filename: str,
        *,
        output_dir: Path = Path("Plots"),
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_dir / filename,
            metadata=None if metadata is None else dict(metadata),
        )
        plt.show()
