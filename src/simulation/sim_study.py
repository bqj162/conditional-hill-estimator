import numpy as np
import pandas as pd
import sys
import time
from typing import Callable, Dict, Iterable, Any
# from src.simulation.simulator import simulate_chain
from .simulator import simulate_chain
from src.estimators.HillEstimator import HillEstimator
from src.plotting.Plots import Plots
from joblib import Parallel, delayed
from itertools import product

def run_grid_generic(sim_one_func,
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

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(call_one)(idx, vals) for idx, vals in enumerate(combos)
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
        gamma_x += (cond_hil)
    
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
        chain = simulate_chain(gamma_func, n = 1, family = family, burn_in = burn_in + 1, rng = rng)
        Z_t = chain[-1]
        z_q = (1-q)**(-gamma_func(Z_t))
        # z_q = (-np.log(1-q))**(-gamma_func(Z_t))
        rv = chain[0:burn_in]
        cv = np.r_[init_state, chain[0:(burn_in-1)]] 
        rv_sorted = np.sort(rv)
        threshold = rv_sorted[burn_in - k_n - 1]

        hill     = HillEstimator.unconditional_hill_vectorized(Y = rv, k_array = k_n)
        cond_hil = HillEstimator.gamma_fixed_k_n_x_vectorized(X = cv, Y = rv, k_array = k_n, x = Z_t)

        #---
        # rv_ex = rv - cond_hil['q_n']
        # hill  = HillEstimator.unconditional_hill_vectorized(Y = rv - threshold, k_array = k_n)
        # cond_hil = HillEstimator.gamma_fixed_k_n_x_vectorized(X = cv, Y = rv_ex, k_array = k_n, x = Z_t)
        # z_hat   = threshold       + (hill)*(((1-q)/(k_n/burn_in))**(-hill)-1)
        # z_hat_t = cond_hil['q_n'] + (cond_hil['gamma'])*(((1-q)/(k_n/burn_in))**(-cond_hil['gamma'])-1)
        #---
            
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
    try:
        import debugpy
        return debugpy.is_client_connected()
    except Exception:
        # fallback for other debuggers or if debugpy isn't installed
        import sys
        return sys.gettrace() is not None


if __name__ == "__main__":
    def gamma(x): return 1/(1+abs(x)) # def gamma(x): return 3*x*(x-1)+1
    t0 = time.perf_counter()
    args = {'n_samples': 100, 'gamma_func' : gamma, 'family': "Pareto", 'init_state': 1, 'base_seed':0}
    burn_ins = [1000, 10000]
    sim_bias_MSE_vec_part(n_samples= args['n_samples'], burn_in= burn_ins[0], gamma_func= args['gamma_func'], family= args['family'], init_state= args['init_state'], q = 0.95, base_seed= args['base_seed'])
    # x_evals = [2, 5]
    # sweep_x = {'x_eval': [2, 5], 'burn_in' : [1000, 10000]}
    # grid_x = run_grid_generic(sim_one_func=sim_bias_MSE_gamma, fixed_args= args, sweep=sweep_x, n_jobs=-2)
    # Plots.plot_2x2_grid_param(results_df =grid_x, burn_ins = burn_ins, col_param ='x_eval', col_values = x_evals)

    q_vals = [.95, .99]
    sweep_q = {'q': q_vals, 'burn_in' : [1000, 10000]}
    grid_q = run_grid_generic(sim_one_func=sim_bias_MSE_vec_part, fixed_args= args, sweep=sweep_q, n_jobs=-2)
    Plots.plot_2x2_grid_param(results_df =grid_q, burn_ins = burn_ins, col_param ='q', col_values = q_vals)

    t1 = time.perf_counter()
    print(f"sim_bias_MSE took {t1 - t0:.3f} seconds")
   





#Original implementation, not vectorised.
# def sim_bias_MSE(n_samples: int, burn_in: int, alpha: Callable[[float], float], init_state: float, q: float) -> pd.DataFrame:

#     k_n = np.unique(np.floor(np.linspace(0.01 * burn_in, 0.5 * burn_in, 200)).astype(int))
#     sum_bias   = np.zeros(len(k_n), dtype=float)
#     sum_bias_t = np.zeros(len(k_n), dtype=float)
#     sum_sq     = np.zeros(len(k_n), dtype=float)
#     sum_sq_t   = np.zeros(len(k_n), dtype=float)

#     for i in range(n_samples):
#         rng = np.random.default_rng(seed=i)
#         chain = simulate_chain(alpha, n = 1, burn_in = burn_in + 1, rng = rng)
#         Z_t = chain[-1]
#         z_q = (1-q)**(-1/alpha(Z_t))
#         z_hat   = np.empty([len(k_n)])
#         z_hat_t = np.empty([len(k_n)])

#         for j in range(k_n.size):
#             k  = k_n[j]
#             rv = chain[0:burn_in]
#             cv = np.r_[init_state, chain[0:(burn_in-1)]] 
#             hill      = HillEstimator.unconditional_hill_estimator(Y = rv, k_n = k)
#             cond_hil  = HillEstimator.gamma_fixed_k_n_x(X = cv, Y = rv, k_n = k, x = Z_t)
#             rv_sorted = np.sort(rv)
#             threshold = rv_sorted[burn_in - k - 1]

#             z_hat[j]   = threshold*((1-q)/(k/burn_in))**(-hill)
#             z_hat_t[j] = threshold*((1-q)/(k/burn_in))**(-cond_hil)
        
#         sum_bias   += (z_hat   - z_q)
#         sum_bias_t += (z_hat_t - z_q) 
#         sum_sq     += (z_hat   - z_q)**2
#         sum_sq_t   += (z_hat_t - z_q)**2

#     return pd.DataFrame({'n': burn_in,'k': k_n, 'bias': sum_bias/n_samples, 'bias_t': sum_bias_t/n_samples, 'MSE': sum_sq/n_samples, 'MSE_t': sum_sq_t/n_samples})




#Very slow
# def sim_bias_MSE_vec(n_samples: int, burn_in: int,
#                      alpha: Callable[[np.ndarray], np.ndarray],
#                      init_state: float, q: float) -> pd.DataFrame:

#     k_n = np.unique(np.floor(np.linspace(0.01 * burn_in, 0.5 * burn_in, 200)).astype(int))
#     m = k_n.size
#     rng = np.random.default_rng(seed=12345)

#     # simulate all chains in parallel
#     chains = simulate_chains(alpha=alpha, n_samples=n_samples, burn_in=burn_in,
#                              init_state=init_state, rng=rng)  # (ns, burn_in+1)

#     rv_mat = chains[:, :burn_in]   # (ns, n)
#     cv_mat = np.concatenate([np.full((n_samples, 1), init_state), chains[:, :burn_in-1]], axis=1)  # (ns, n)
#     Z_t = chains[:, -1]            # (ns,)

#     # true quantile per sample
#     z_q = (1.0 - q) ** (-1.0 / alpha(Z_t))     # (ns,)

#     # thresholds (ns, m): Y_sorted[:, n - k - 1]
#     Y_sorted = np.sort(rv_mat, axis=1)         # (ns, n)
#     cols = burn_in - k_n - 1                   # shape (m,)
#     row_idx = np.arange(n_samples)[:, None]
#     thresholds = Y_sorted[row_idx, cols[None, :]]   # (ns, m)

#     # base term (m,) -> (1, m) later
#     base = (1.0 - q) / (k_n / burn_in)               # (m,)

#     # unconditional hill: (ns, m)
#     hill_mat = HillEstimator.batch_unconditional_hill(rv_mat, k_n)  # (ns, m)

#     # conditional hill: (ns, m) - heavy memory
#     cond_hil_mat = HillEstimator.batch_gamma_fixed_k_n_x(cv_mat, rv_mat, k_n, Z_t)  # (ns, m)

#     # predictions per sample and k: (ns, m)
#     z_hat_mat   = thresholds * (base[None, :] ** (-hill_mat))
#     z_hat_t_mat = thresholds * (base[None, :] ** (-cond_hil_mat))

#     # differences per sample,k
#     delta_mat   = z_hat_mat   - z_q[:, None]   # (ns, m)
#     delta_t_mat = z_hat_t_mat - z_q[:, None]   # (ns, m)

#     # aggregate over samples (sum then divide by n_samples)
#     sum_bias = delta_mat.sum(axis=0)           # (m,)
#     sum_bias_t = delta_t_mat.sum(axis=0)       # (m,)
#     sum_sq = (delta_mat ** 2).sum(axis=0)      # (m,)
#     sum_sq_t = (delta_t_mat ** 2).sum(axis=0)  # (m,)

#     n = n_samples
#     bias = sum_bias / n
#     bias_t = sum_bias_t / n
#     mse = sum_sq / n
#     mse_t = sum_sq_t / n

#     return pd.DataFrame({
#         'n': np.repeat(burn_in, m),
#         'k': k_n,
#         'bias': bias,
#         'bias_t': bias_t,
#         'MSE': mse,
#         'MSE_t': mse_t
#     })


#As fast as when the inner loop is vectorised 
# def sim_bias_MSE_chunked(n_samples: int, burn_in: int, alpha: Callable[[np.ndarray], np.ndarray],
#                          init_state: float, q: float, batch_size: int = 64) -> pd.DataFrame:
#     """
#     Chunked simulation to reproduce per-sample seeding (seed=i) but with vectorized
#     computation inside each chunk of size B. Returns DataFrame same shape as before.
#     """
#     k_n = np.unique(np.floor(np.linspace(0.01 * burn_in, 0.5 * burn_in, 200)).astype(int))
#     m = k_n.size

#     sum_bias = np.zeros(m, dtype=float)
#     sum_bias_t = np.zeros(m, dtype=float)
#     sum_sq = np.zeros(m, dtype=float)
#     sum_sq_t = np.zeros(m, dtype=float)

#     # process samples in chunks to limit memory
#     for start in range(0, n_samples, batch_size):
#         end = min(start + batch_size, n_samples)
#         B = end - start

#         # Build U matrix for this chunk so that seed for sample i is exactly i
#         # This replicates rng = np.random.default_rng(seed=i) used originally
#         U_chunk = np.vstack([np.random.default_rng(seed=i).random(burn_in) for i in range(start, end)])
#         # Simulate chains for chunk using vectorized recurrence
#         X = np.empty((B, burn_in + 1), dtype=float)
#         X[:, 0] = init_state
#         for t in range(1, burn_in + 1):
#             a = alpha(X[:, t-1])            # vectorized alpha over chunk (shape (B,))
#             X[:, t] = U_chunk[:, t-1] ** (-1.0 / a)

#         rv_mat = X[:, :burn_in]            # shape (B, burn_in)
#         cv_mat = np.concatenate([np.full((B, 1), init_state), X[:, :burn_in-1]], axis=1)  # (B,n)
#         Z_t = X[:, -1]                     # (B,)

#         # True quantile per sample in chunk
#         z_q_chunk = (1.0 - q) ** (-1.0 / alpha(Z_t))   # (B,)

#         # thresholds per sample & k
#         Y_sorted = np.sort(rv_mat, axis=1)            # (B,n)
#         cols = burn_in - k_n - 1                      # (m,)
#         thresholds = Y_sorted[np.arange(B)[:, None], cols[None, :]]  # (B,m)

#         base = (1.0 - q) / (k_n / burn_in)            # (m,)

#         # compute hill and cond_hil for chunk
#         hill_chunk = HillEstimator.batch_unconditional_hill(rv_mat, k_n)   # (B,m) - vectorized over rows
#         cond_hil_chunk = HillEstimator.batch_gamma_fixed_k_n_x(cv_mat, rv_mat, k_n, Z_t)  # (B,m)

#         z_hat_chunk   = thresholds * (base[None, :] ** (-hill_chunk))     # (B,m)
#         z_hat_t_chunk = thresholds * (base[None, :] ** (-cond_hil_chunk)) # (B,m)

#         delta = z_hat_chunk - z_q_chunk[:, None]           # (B,m)
#         delta_t = z_hat_t_chunk - z_q_chunk[:, None]       # (B,m)

#         sum_bias += delta.sum(axis=0)
#         sum_bias_t += delta_t.sum(axis=0)
#         sum_sq += (delta ** 2).sum(axis=0)
#         sum_sq_t += (delta_t ** 2).sum(axis=0)

#     n = n_samples
#     bias = sum_bias / n
#     bias_t = sum_bias_t / n
#     mse = sum_sq / n
#     mse_t = sum_sq_t / n

#     return pd.DataFrame({
#         'n': np.repeat(burn_in, m),
#         'k': k_n,
#         'bias': bias,
#         'bias_t': bias_t,
#         'MSE': mse,
#         'MSE_t': mse_t
#     })



# def run_grid(sim_one_func, burn_ins, x_evals, n_jobs=1, **sim_kwargs):
#     combos = list(product(burn_ins, x_evals))
    
#     results = Parallel(n_jobs=n_jobs, backend='threading')(
#         delayed(sim_one_func)(sim_kwargs['n_samples'], b, sim_kwargs['gamma_func'],
#                               sim_kwargs['init_state'], x, sim_kwargs.get('base_seed',0))
#         for b, x in combos
#     )
#     tidy = pd.concat(results, ignore_index=True)
#     return tidy