import numpy as np
import math
from typing import Callable, Optional

def simulate_chain(
    alpha: Callable[[float], float],
    n: int,
    X0: float = 1.0,
    burn_in: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate a single Markov chain X_j = U_j^{-1/alpha(X_{j-1})} for n steps.
    - alpha: function alpha(x) -> positive float
    - n: number of returned samples
    - X0: initial state (if you have invariant dist sample, pass it)
    - burn_in: number of burn-in steps before collecting samples
    - rng: optional np.random.Generator
    """
    if rng is None:
        rng = np.random.default_rng()

    x = float(X0)

    for _ in range(burn_in):
        u = float(rng.random())
        x = math.exp(-math.log(u) / float(alpha(x)))

    out = np.empty(n, dtype=float)
    for i in range(n):
        u = float(rng.random())
        x = math.exp(-math.log(u) / float(alpha(x)))
        out[i] = x

    return out


if __name__ == "__main__":
    def alpha(x): return 1 + abs(x)
    samples = simulate_chain(alpha, n=10000, X0=1.0, burn_in=1000)
    print(f"Mean: {np.mean(samples)}, Std: {np.std(samples)}")

