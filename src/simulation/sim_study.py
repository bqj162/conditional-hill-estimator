import numpy as np
import pandas as pd
import sys
import time
from typing import Callable, Dict, Iterable, Any, cast
# from src.simulation.simulator import simulate_chain
from .simulator import simulate_chain
from src.estimators.HillEstimator import HillEstimator
from src.plotting.Plots import Plots
from joblib import Parallel, delayed
from itertools import product

def run_grid_generic(sim_one_func: Callable[..., pd.DataFrame],
                     fixed_args: Dict[str, Any],
                     sweep: Dict[str, Iterable],
                     n_jobs: int = 1,
                     base_seed: int = 0) -> pd.DataFrame:
    """
    sim_one_func: function that runs one scenario and returns a DataFrame.
      It must accept named args (e.g. burn_in=..., x_eval=..., q=..., base_seed=...).
    fixed_args: dict of args that stay fixed across runs (e.g. {'n_samples': 200, 'gamma_func': gamma_2, 'init_state': 1})
    sweep: dict mapping parameter-name -> iterable of values you want to sweep (e.g. {'burn_in':[100,500], 'q':[0.95,0.99]})
    n_jobs: parallel workers (use 1 while debugging)
    """

    # all names and value lists
    keys = list(sweep.keys())
    combos = list(product(*[sweep[k] for k in keys]))
    debugging = _is_debugging()
    backend = 'threading' if debugging else 'loky'

    def call_one(combo_idx, combo_values):
        # build kwargs for this run
        kwargs = dict(fixed_args)
        for k, v in zip(keys, combo_values):
            kwargs[k] = v
        # make base_seed deterministic per combo and optionally per-repetition inside sim_one
        kwargs['base_seed'] = base_seed + combo_idx
        return sim_one_func(**kwargs)

    results = cast(
        list[pd.DataFrame],
        Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(call_one)(idx, vals) for idx, vals in enumerate(combos)
        ),
    )
    return pd.concat(results, ignore_index=True)

def sim_gamma(n_samples: int, burn_in: int, gamma_func: Callable[[float], float], init_state: float, x_eval: float) -> pd.DataFrame:

    k_n = np.unique(np.floor(np.linspace(0.01 * burn_in, 0.5 * burn_in, 200)).astype(int))
    gamma   = np.zeros(len(k_n), dtype=float)
    gamma_x = np.zeros(len(k_n), dtype=float)
    target = gamma_func(x_eval)

    for i in range(n_samples):
        rng = np.random.default_rng(seed=i)
        chain = simulate_chain(gamma_func, n = 1, burn_in = burn_in + 1, rng = rng)
        rv = chain[0:burn_in]
        cv = np.r_[init_state, chain[0:(burn_in-1)]]
        hill     = HillEstimator.unconditional_hill_vectorized(Y = rv, k_array = k_n)
        cond_hil = HillEstimator.gamma_fixed_k_n_x_vectorized(X = cv, Y = rv, k_array = k_n, x = x_eval)
        gamma   += (hill)
        gamma_x += cond_hil['gamma'].to_numpy()

    return pd.DataFrame({'k': k_n, 'gamma': gamma/n_samples, 'gamma_x': gamma_x/n_samples, 'target' : target})


def sim_bias_MSE_gamma(n_samples: int, burn_in: int, gamma_func: Callable[[float], float], family:str, init_state: float, x_eval: float, base_seed:int = 0) -> pd.DataFrame:

    k_n = np.unique(np.floor(np.linspace(0.01 * burn_in, .9 * burn_in, 200)).astype(int))
    sum_bias, sum_bias_t, sum_sq, sum_sq_t = [np.zeros(len(k_n), dtype=float) for _ in range(4)]
    target = gamma_func(x_eval)

    for i in range(n_samples):
        rng = np.random.default_rng(seed=base_seed+i)
        chain = simulate_chain(gamma_func, n = 1, family = family, burn_in = burn_in + 1, rng = rng)
        rv = chain[0:burn_in]
        cv = np.r_[init_state, chain[0:(burn_in-1)]]
        hill     = HillEstimator.unconditional_hill_vectorized(Y = rv, k_array = k_n)
        cond_hil = HillEstimator.gamma_fixed_k_n_x_vectorized(X = cv, Y = rv, k_array = k_n, x = x_eval)

        sum_bias   += (hill     - target)
        sum_bias_t += (cond_hil['gamma'] - target)
        sum_sq     += (hill     - target)**2
        sum_sq_t   += (cond_hil['gamma'] - target)**2

    return pd.DataFrame({'burn_in': burn_in,'x_eval': x_eval,'k': k_n,
                         'bias': sum_bias/n_samples, 'bias_t': sum_bias_t/n_samples,
                         'MSE': sum_sq/n_samples, 'MSE_t': sum_sq_t/n_samples})


def sim_bias_MSE_vec_part(n_samples: int, burn_in: int, gamma_func: Callable[[float], float], family:str, init_state: float, q: float, base_seed:int = 0) -> pd.DataFrame:

    k_n = np.unique(np.floor(np.linspace((1-q) * burn_in, 0.9 * burn_in, 200)).astype(int))
    sum_bias, sum_bias_t, sum_sq, sum_sq_t = [np.zeros(len(k_n), dtype=float) for _ in range(4)]

    for i in range(n_samples):
        rng = np.random.default_rng(seed=base_seed+i)
        chain = simulate_chain(gamma_func, n = 1, family = family, burn_in = burn_in + 1, rng = rng, X0=init_state)
        Z_t = chain[-1]
        # z_q = (1-q)**(-gamma_func(Z_t))
        z_q = (-np.log(q))**(-gamma_func(Z_t))
        rv = chain[0:burn_in]
        cv = np.r_[init_state, chain[0:(burn_in-1)]]
        rv_sorted = np.sort(rv)
        threshold = rv_sorted[burn_in - k_n - 1]

        hill     = HillEstimator.unconditional_hill_vectorized(Y = rv, k_array = k_n)
        cond_hil = HillEstimator.gamma_fixed_k_n_x_vectorized(X = cv, Y = rv, k_array = k_n, x = Z_t)

        z_hat   = threshold*((1-q)/(k_n/burn_in))**(-hill)
        z_hat_t = cond_hil['q_n']*((1-q)/(k_n/burn_in))**(-cond_hil['gamma'])

        sum_bias   += (z_hat   - z_q)
        sum_bias_t += (z_hat_t - z_q)
        sum_sq     += (z_hat   - z_q)**2
        sum_sq_t   += (z_hat_t - z_q)**2

    return pd.DataFrame({'burn_in': burn_in, 'q': q, 'k': k_n,
                         'bias': sum_bias/n_samples, 'bias_t': sum_bias_t/n_samples,
                         'MSE': sum_sq/n_samples, 'MSE_t': sum_sq_t/n_samples})

def _is_debugging():
    return sys.gettrace() is not None


if __name__ == "__main__":
    def gamma(x): return 1/(1+1/x) # def gamma(x): return 3*x*(x-1)+1
    t0 = time.perf_counter()
    # chain = simulate_chain(gamma, n = 1, family = "Frechet", burn_in = 100 + 1, rng = np.random.default_rng(seed=1))
    args = {'n_samples': 100, 'gamma_func' : gamma, 'family': "Frechet", 'init_state': 1, 'base_seed':0}
    burn_ins = [1000, 10000]
    # sim_bias_MSE_vec_part(n_samples= args['n_samples'], burn_in= burn_ins[0], gamma_func= args['gamma_func'], family= args['family'], init_state= args['init_state'], q = 0.95, base_seed= args['base_seed'])
    # x_evals = [2, 5]
    # sweep_x = {'x_eval': [2, 5], 'burn_in' : [1000, 10000]}
    # grid_x = run_grid_generic(sim_one_func=sim_bias_MSE_gamma, fixed_args= args, sweep=sweep_x, n_jobs=-2)
    # Plots.plot_2x2_grid_param(results_df=grid_x, burn_ins=burn_ins,
    #                           col_param="x_eval", col_values=x_evals,
    #                           family=args["family"])

    q_vals = [.95, .99]
    sweep_q = {'q': q_vals, 'burn_in' : [1000, 10000]}
    grid_q = run_grid_generic(sim_one_func=sim_bias_MSE_vec_part, fixed_args= args, sweep=sweep_q, n_jobs=-2)
    Plots.plot_2x2_grid_param(
        results_df=grid_q,
        burn_ins=burn_ins,
        col_param="q",
        col_values=q_vals,
        family=args["family"],
    )

    t1 = time.perf_counter()
    print(f"sim_bias_MSE took {t1 - t0:.3f} seconds")
