import unittest

import numpy as np

from src.data.TimeSeries import TimeSeries


class TimeSeriesTests(unittest.TestCase):
    def test_split_returns_positive_tail_magnitudes_and_excludes_zeros(self) -> None:
        series = TimeSeries(
            time=np.arange(5),
            covariate_name="lagged_return",
            covariate=np.arange(5, dtype=float),
            rv_name="return",
            rv=np.array([-2.0, 0.0, 1.0, -0.5, 3.0]),
        )

        positive, negative = series.split()

        np.testing.assert_array_equal(positive.rv, np.array([1.0, 3.0]))
        np.testing.assert_array_equal(negative.rv, np.array([2.0, 0.5]))
        self.assertTrue(np.all(positive.rv > 0))
        self.assertTrue(np.all(negative.rv > 0))

    def test_unsplit_series_represents_losses(self) -> None:
        series = TimeSeries(rv_name="return", rv=np.array([0.1, -0.2]))

        losses = series.split(split=False)[0]

        self.assertEqual(losses.rv_name, "return_loss")
        np.testing.assert_array_equal(losses.rv, np.array([-0.1, 0.2]))

    def test_constructor_rejects_misaligned_arrays(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            TimeSeries(rv_name="return", rv=[1.0, 2.0], time=[0])


if __name__ == "__main__":
    unittest.main()
