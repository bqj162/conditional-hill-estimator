import unittest

import numpy as np

from src.simulation.simulator import simulate_chain


class SimulatorTests(unittest.TestCase):
    @staticmethod
    def gamma(_: float) -> float:
        return 0.5

    def test_seeded_simulation_is_reproducible(self) -> None:
        first = simulate_chain(
            self.gamma,
            family="Frechet",
            n=1,
            burn_in=20,
            rng=np.random.default_rng(7),
        )
        second = simulate_chain(
            self.gamma,
            family="frechet",
            n=1,
            burn_in=20,
            rng=np.random.default_rng(7),
        )

        np.testing.assert_array_equal(first, second)

    def test_unknown_family_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "family"):
            simulate_chain(self.gamma, family="lognormal", n=1, burn_in=2)


if __name__ == "__main__":
    unittest.main()
