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
    if rng is None:
        rng = np.random.default_rng()

    x = float(X0)

    results = np.empty([n, burn_in])
    for i in range(n):
        x = float(X0)
        for j in range(burn_in):
            u = float(rng.random())
            x = math.exp(-math.log(u) / float(alpha(x)))
            results[i,j] = x

    if n == 1:
        return results[0]
    return results

    


if __name__ == "__main__":
    def alpha(x): return 1 + abs(x)
    samples = simulate_chain(alpha, n=10, X0=1.0, burn_in=5)
    print(samples)
    print(f"Mean: {np.mean(samples)}, Std: {np.std(samples)}")

