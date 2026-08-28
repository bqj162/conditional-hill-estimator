from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


class TimeSeries:
    """Container for a response series and an optional aligned covariate."""

    def __init__(
        self,
        rv_name: str,
        rv: ArrayLike,
        time: ArrayLike | None = None,
        covariate_name: str | None = None,
        covariate: ArrayLike | None = None,
    ) -> None:
        self.time = None if time is None else np.asarray(time)
        self.covariate = None if covariate is None else np.asarray(covariate)
        self.covariate_name = covariate_name
        self.rv = np.asarray(rv)
        self.rv_name = rv_name

        n = self.rv.size
        if self.time is not None and self.time.size != n:
            raise ValueError("time and rv must have the same length")
        if self.covariate is not None and self.covariate.size != n:
            raise ValueError("covariate and rv must have the same length")

    def split(self, split: bool = True) -> list[TimeSeries]:
        """Split conditional data into positive and negative response tails.

        With split=False, return the loss series -rv used by the one-sided
        financial forecasting pipeline.
        """
        if not split:
            return [
                TimeSeries(
                time=self.time,
                rv_name=f"{self.rv_name}_loss",
                rv=-self.rv,
            )
        ]

        if self.covariate is None or self.covariate_name is None:
            raise ValueError("A covariate is required when split=True")

        positive = self.rv > 0
        negative = self.rv < 0
        tails: list[TimeSeries] = []
        if np.any(positive):
            tails.append(self._select_tail(positive, flip=False))
        if np.any(negative):
            tails.append(self._select_tail(negative, flip=True))
        if not tails:
            raise ValueError("Cannot estimate tails from an all-zero response series")
        return tails

    def transform(self, transform_type: str | None) -> None:
        if transform_type is None:
            return
        if transform_type != "log_diff":
            raise ValueError(f"Unknown transform type: {transform_type}")
        self.log_difference()

    def log_difference(self) -> None:
        if np.any(self.rv.astype(float) <= 0):
            raise ValueError("log_diff requires strictly positive response values")
        if self.covariate is not None:
            if np.any(self.covariate.astype(float) <= 0):
                raise ValueError("log_diff requires strictly positive covariate values")
            self.covariate = np.diff(np.log(self.covariate.astype(float)))
        self.rv = np.diff(np.log(self.rv.astype(float)).squeeze())
        if self.time is not None:
            self.time = self.time[1:]

    def _select_tail(self, index: NDArray[np.bool_], flip: bool) -> TimeSeries:
        suffix = "negative" if flip else "positive"
        covariate = None if self.covariate is None else self.covariate[index]
        return TimeSeries(
            time=None if self.time is None else self.time[index],
            covariate_name=self.covariate_name,
            covariate=covariate,
            rv_name=f"{self.rv_name}_{suffix}",
            rv=self.rv[index] * (-1 if flip else 1),
        )

    def to_rv_series(self) -> pd.Series:
        return pd.Series(index=self.time, data=self.rv, name=self.rv_name)
