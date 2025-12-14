import numpy as np
import math
from typing import Callable, Optional

def simulate_chain(
    gamma: Callable[[float], float],
    family: str = "Pareto",
    n: int = 1000,
    X0: float = 1.0,
    burn_in: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    x = float(X0)

    results = np.empty([n, burn_in])
    if family == "Pareto":
        for i in range(n):
            x = float(X0)
            for j in range(burn_in):
                u = float(rng.random())
                # x = math.exp(-math.log(u) / float(alpha(x)))
                x = (1-u)**(-gamma(x))
                results[i,j] = x
    if family == "Frechet":
        for i in range(n):
            x = float(X0)
            for j in range(burn_in):
                u = float(rng.random())
                x = (-np.log(1-u))**(-gamma(x)) 
                results[i,j] = x

    if n == 1:
        return results[0]
    return results

def simulate_chains(alpha: Callable[[np.ndarray], np.ndarray],
                    n_samples: int,
                    burn_in: int,
                    init_state: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Simulate n_samples independent chains in parallel.
    Returns array shape (n_samples, burn_in+1) where column 0 is X_0 = init_state.
    alpha must accept and return numpy arrays.
    """
    n_steps = burn_in
    U = rng.random((n_samples, n_steps))          # uniforms for steps 1..burn_in
    X = np.empty((n_samples, n_steps + 1), dtype=float)
    X[:, 0] = init_state
    for t in range(1, n_steps + 1):
        a = alpha(X[:, t-1])                      # vectorized alpha over samples
        X[:, t] = U[:, t-1] ** (-1.0 / a)
    return X
    


if __name__ == "__main__":
    def alpha(x): return 1 + abs(x)
    samples = simulate_chain(alpha, n=10, X0=1.0, burn_in=5)
    print(samples)
    print(f"Mean: {np.mean(samples)}, Std: {np.std(samples)}")

