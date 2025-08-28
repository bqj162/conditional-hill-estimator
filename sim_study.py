import numpy as np
import pandas as pd
from typing import Callable
from simulator import simulate_chain 
from HillEstimator import HillEstimator
from Plots import Plots

def sim_bias_MSE(n_samples: int, burn_in: int, alpha: Callable[[float], float], init_state: float, q: float) -> pd.DataFrame:

    k_n = np.unique(np.floor(np.linspace(0.01 * burn_in, 0.5 * burn_in, burn_in)).astype(int))
    sum_bias   = np.zeros(len(k_n), dtype=float)
    sum_bias_t = np.zeros(len(k_n), dtype=float)

    for i in range(n_samples):
        rng = np.random.default_rng(seed=i)
        chain = simulate_chain(alpha, n = 1, burn_in = burn_in + 1, rng = rng)
        Z_t = chain[-1]
        z_q = (1-q)**(-1/alpha(Z_t))
        z_hat   = np.empty([len(k_n)])
        z_hat_t = np.empty([len(k_n)])

        for i in range(k_n.size):
            k  = k_n[i]
            rv = chain[0:burn_in]
            cv = np.r_[init_state, chain[0:(burn_in-1)]] 
            hill      = HillEstimator.unconditional_hill_estimator(Y = rv, k_n = k)
            cond_hil  = HillEstimator.gamma_fixed_k_n_x(X = cv, Y = rv, k_n = k, x = Z_t)
            rv_sorted = np.sort(rv)
            threshold = rv_sorted[burn_in - k - 1]

            z_hat[i]   = threshold*((1-q)/(k/burn_in))**(-hill)
            z_hat_t[i] = threshold*((1-q)/(k/burn_in))**(-cond_hil)
        
        sum_bias   += (z_hat   - z_q)
        sum_bias_t += (z_hat_t - z_q) 

    return pd.DataFrame({'n': burn_in,'k': k_n, 'bias': sum_bias/len(k_n), 'bias_t': sum_bias_t/len(k_n), 'MSE': (sum_bias**2)/len(k_n), 'MSE_t': (sum_bias_t**2)/len(k_n)})

if __name__ == "__main__":
    def alpha(x): return 1 + abs(x)
    sim_1 = sim_bias_MSE(100, burn_in=2000, alpha = alpha, init_state= 1, q= .95)
    Plots.plot_bias_MSE(sim_1)
    







