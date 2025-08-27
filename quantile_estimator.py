import pandas as pd
import numpy as np
from ts_fitting import AR1GARCG11
from HillEstimator import HillEstimator
from TimeSeries import TimeSeries

class quantileSeries:
    def __init__(self, time_series, q, fitting_window):
        self.time_series = time_series
        self.q = q
        self.fitting_window = fitting_window
        self.quantile_series = None

    def estimate(self):
        self.quantile_series = self.quantile_all_dates(time_series=self.time_series, 
                                                       fitting_window=self.fitting_window)
        return self.quantile_series
    
    def quantile_all_dates(self, time_series, fitting_window: int):
        n = len(time_series.rv)
        if n < fitting_window:
            raise Exception("Fitting window must be smaller than length of time series")
        results = []
        for i in range(fitting_window, n-1):
            ts_slice = time_series.time[ i - fitting_window : i ] 
            vals_slice = time_series.rv[ i - fitting_window : i ]
            window_series = TimeSeries(time=ts_slice, rv=vals_slice, rv_name=time_series.rv_name)

            quantiles = self.quantile_at_t(window_series)
            quantiles["obs"] = time_series.rv[i+1]
            results.append(quantiles)
        out = pd.concat(results, ignore_index=True)
        out.set_index('date', inplace=True)
        return out
        
    
    def quantile_at_t(self, time_series):
        returns = time_series.to_rv_series()
        model = AR1GARCG11.fit(returns=returns, name = time_series.rv_name)
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
        Hill = HillEstimator(pos_ts)
        n = len(Hill.time_series.rv)
        # k_n = int(np.floor(n/10)) 
        # k_n = int(np.floor(n/20)) 
        # k_n = int(np.floor(np.sqrt(n)))
        # k_n = int(np.floor(n**(3/5)))
        k_n = int(np.floor(n**(2/3)))
        # k_n = int(np.floor(n**(3/4)))
        # k_n = int(np.floor(n**(4/5)))
        gamma = Hill.gamma_fixed_k_n_x(X = Hill.time_series.covariate,
                                       Y = Hill.time_series.rv, 
                                       k_n = int(k_n), 
                                       x = z_t)
        
        gamma_unc = Hill.unconditional_hill_estimator(Y = Hill.time_series.rv, k_n=k_n)
        gains_sorted = np.sort(pos_ts.rv)
        order_stat   = gains_sorted[-(k_n + 1)]
        
        # exceendances              = gains_sorted[-k_n:] - order_stat
        # c_hat, loc_hat, scale_hat = genpareto.fit(exceendances, floc=0)    
        # z_hat_gpd                 = order_stat + (scale_hat/c_hat) * (((1 - self.q)/(k_n/n))**(-c_hat) - 1)

        out = pd.DataFrame({
            "date": [fitting.forecast_date for q in self.q],
            "q": self.q,
            "x_hat": [fitting.forecast_mu + fitting.forecast_sigma *
                    order_stat*((1-q)/(k_n/n))**(-gamma) for q in self.q],
            "x_hat_unc": [fitting.forecast_mu + fitting.forecast_sigma *
                        order_stat*((1-q)/(k_n/n))**(-gamma_unc) for q in self.q],
            "obs": None
        })
        return out
