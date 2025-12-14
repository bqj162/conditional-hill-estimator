import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import cProfile, pstats
from .ts_fitting import AR1GARCG11
from .HillEstimator import HillEstimator
from src.data.TimeSeries import TimeSeries

class quantileSeries:
    def __init__(self, time_series, q, fitting_window):
        self.time_series = time_series
        self.q = q
        self.fitting_window = fitting_window
        self.quantile_series = None
        self.start_values = None

    def estimate(self):
        # pr = cProfile.Profile()
        # pr.enable()
        self.quantile_series = self.quantile_all_dates(time_series=self.time_series, fitting_window=self.fitting_window)
        # pr.disable()
        # ps = pstats.Stats(pr).sort_stats('tottime')
        # ps.print_stats(30)
        return self.quantile_series
        

    def quantile_all_dates(self, time_series, fitting_window: int):
        n = len(time_series.rv)
        if n < fitting_window:
            raise Exception("Fitting window must be smaller than length of time series")

        results = []
        start_values = None  
        for i in range(fitting_window, n-1):
            ts_slice = np.asarray(time_series.time[i - fitting_window : i])
            vals_slice = np.asarray(time_series.rv[i - fitting_window : i])

            window_series = TimeSeries(time=ts_slice,rv=vals_slice,rv_name=time_series.rv_name)
            quantiles, fitted_model = self.quantile_at_t(window_series, start_values=start_values)

            start_values = fitted_model
            quantiles["obs"] = time_series.rv[i+1]
            results.append(quantiles)

        out = pd.concat(results, ignore_index=True)
        out.set_index("date", inplace=True)
        return out
    
    def quantile_at_t(self, time_series, start_values):
        returns = time_series.to_rv_series()
        model = AR1GARCG11.fit(returns=returns, name = time_series.rv_name, start_values=start_values) #, fit_kwargs={'method':'BFGS', 'options':{'maxiter':200}}
        self.start_values = model.fitted_model
        fitting = model.forecast_1()

        ts = fitting.residual_ts
        mask = ts.rv > 0
        pos_time      = ts.time[mask]
        pos_covariate = ts.covariate[mask]
        pos_rv        = ts.rv[mask]

        pos_ts = TimeSeries(
            time           = pos_time,
            covariate_name = ts.covariate_name,
            covariate      = pos_covariate,
            rv_name        = ts.rv_name,
            rv             = pos_rv
        )
        z_t = pos_rv[-1]
        n = len(pos_ts.rv)
        # k_n = int(np.floor(n/20)) 
        # k_n = int(np.floor(n/10)) 
        # k_n = int(np.floor(np.sqrt(n)))
        # k_n = int(np.floor(n**(3/5)))
        k_n = int(np.floor(n**(2/3)))
        # k_n = int(np.floor(n**(3/4)))
        # k_n = int(np.floor(n**(4/5)))
        # gamma = Hill.gamma_fixed_k_n_x(X = Hill.time_series.covariate,
        #                                Y = Hill.time_series.rv, 
        #                                k_n = int(k_n), 
        #                                x = z_t)
        gamma, q_n = HillEstimator.gamma_fixed_k_n_x(X = pos_ts.covariate,
                                       Y = pos_ts.rv, 
                                       k_n = int(k_n), 
                                       x = z_t)
        
        gamma_unc = HillEstimator.unconditional_hill_estimator(Y = pos_ts.rv, k_n=k_n)
        # gains_sorted = np.sort(pos_ts.rv)
        # order_stat   = gains_sorted[-(k_n + 1)]
        order_stat = np.partition(pos_ts.rv, -k_n-1)[-k_n-1]
         
        # exceendances              = gains_sorted[-k_n:] - order_stat
        # c_hat, loc_hat, scale_hat = genpareto.fit(exceendances, floc=0)    
        # z_hat_gpd                 = order_stat + (scale_hat/c_hat) * (((1 - self.q)/(k_n/n))**(-c_hat) - 1)

        out = pd.DataFrame({
            "date": [fitting.forecast_date for q in self.q],
            "q": self.q,
            "x_hat": [fitting.forecast_mu + fitting.forecast_sigma *
                    q_n*((1-q)/(k_n/n))**(-gamma) for q in self.q],
            "x_hat_unc": [fitting.forecast_mu + fitting.forecast_sigma *
                        order_stat*((1-q)/(k_n/n))**(-gamma_unc) for q in self.q],
            "obs": None
        })
        return out, model.fitted_model




    # def quantile_all_dates(self, n_jobs: int = 1) -> pd.DataFrame:
    #     rv = np.asarray(self.time_series.rv)
    #     times = np.asarray(self.time_series.time)
    #     n = rv.shape[0]
    #     fw = self.fitting_window
    #     if n < fw:
    #         raise ValueError("Fitting window must be smaller than length of time series")
    #     # build indices for windows: for i in fw..n-2 inclusive (same as original)
    #     indices = list(range(fw, n-1))

    #     # prepare arguments for each window (pass minimal arrays for speed)
    #     tasks = []
    #     for i in indices:
    #         start = i - fw
    #         end = i  # slice [start:end)
    #         timeslice = times[start:end]
    #         rvslice = rv[start:end]
    #         next_obs = rv[i+1]  # original code stored obs = time_series.rv[i+1]
    #         tasks.append((timeslice, rvslice, next_obs))

    #     # choose parallel or serial
    #     if n_jobs == 1:
    #         results = [self._quantile_at_t_task(t) for t in tasks]
    #     else:
    #         # use threading backend for safety during interactive debug; set n_jobs > 1 for speed
    #         results = Parallel(n_jobs=n_jobs, backend='loky')(
    #             delayed(self._quantile_at_t_task)(t) for t in tasks
    #         )

    #     # results is a list of DataFrames (one per window). concat once.
    #     out = pd.concat(results, ignore_index=True)
    #     out.set_index('date', inplace=True)
    #     return out