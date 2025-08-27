import pandas as pd
import warnings
from arch import arch_model
from arch.univariate.base import ConvergenceWarning
from typing import NamedTuple
from dataclasses import dataclass
from TimeSeries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt 

class Forecast(NamedTuple):
    residual_ts: TimeSeries
    forecast_date: pd.Timestamp
    forecast_mu: float
    forecast_sigma: float

@dataclass
class AR1GARCG11:
    phi: float
    alpha_0: float
    alpha_1: float
    beta: float
    fitted_model: any
    returns: pd.Series
    series_name: str = "resid"

    @classmethod
    def fit(cls, returns: pd.Series, name: str="resid") -> "AR1GARCG11":

        x_t = returns.values
        x_tm1 = returns.shift(1).dropna().values
        x_t = returns.iloc[1:].values 
        phi = float(np.dot(x_t, x_tm1) / np.dot(x_tm1, x_tm1))
        eps = returns.iloc[1:] - phi * returns.shift(1).dropna()
        eps.index = returns.index[1:]

        am = arch_model(
            eps,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            rescale=False
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                res = am.fit(disp="off")
        except ConvergenceWarning:
            res = am.fit(disp="off", method="powell")
        params = res.params

        alpha_0 = float(params.get("omega", params.get("alpha0", None)))
        alpha_1 = float(params.get("alpha[1]", params.get("alpha1", None)))
        beta    = float(params.get("beta[1]", params.get("beta1", None)))

        return cls(
            phi          = phi,
            alpha_0      = alpha_0,
            alpha_1      = alpha_1,
            beta         = beta,
            fitted_model = res,
            returns      = returns,    
            series_name  = name
        )

    
    def forecast_1(self):
        f = self.fitted_model.forecast(horizon=1, reindex=False)
        
        var1 = f.variance.iloc[-1,0] 
        orig_ts = self.returns

        mu_1 = self.phi * orig_ts.values[-1]

        dates = orig_ts.index
        std_resid = self.fitted_model.std_resid.dropna()
        cov_values = std_resid.values[:-1]
        rv_values = std_resid.values[1:]
        time_idx   = std_resid.index         # corresponds to residuals at t = 2,…,n
        time_vals  = time_idx[1:] 
          
        ts = TimeSeries(
            time= time_vals,
            covariate_name = self.series_name + "_lag1",
            covariate=cov_values,
            rv_name=self.series_name ,
            rv= rv_values
        )

        last_date = dates[-1]
        next_date = pd.to_datetime(last_date) + pd.tseries.offsets.BDay(1)  
        
        return Forecast(residual_ts= ts, forecast_date = next_date, forecast_mu=mu_1, forecast_sigma=np.sqrt(var1))
    
