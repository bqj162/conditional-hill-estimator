import pandas as pd
import warnings
from arch import arch_model
from arch.univariate.base import ConvergenceWarning
from typing import NamedTuple
from dataclasses import dataclass
from TimeSeries import TimeSeries
import numpy as np

class Forecast(NamedTuple):
    residual_ts: TimeSeries
    forecast_date: pd.Timestamp
    forecast_mu: float
    forecast_sigma2: float

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

        # am = arch_model(returns, mean="AR",lags=1,vol="GARCH", p=1, q=1, rescale=False)
        # res = am.fit(disp="off")
        # params = res.params

        # ar_keys = [
        #     k for k in params.index
        #     if k.endswith("[1]")
        #        and not k.startswith("alpha")
        #        and not k.startswith("beta")
        # ]
        # if len(ar_keys) != 1:
        #     raise ValueError(f"Could not uniquely identify AR(1) param among {params.index.tolist()}")
        # ar_key = ar_keys[0]
        # phi    = float(params[ar_key])

        # alpha_0 = float(params["omega"])
        # alpha_1 = float(params["alpha[1]"])
        # beta    = float(params["beta[1]"])

        # Step 1: OLS estimate of phi with no intercept
        x_t = returns.values
        x_tm1 = returns.shift(1).dropna().values
        x_t = returns.iloc[1:].values  # align lengths
        # phi = sum(x_{t} * x_{t-1}) / sum(x_{t-1}^2)
        phi = float(np.dot(x_t, x_tm1) / np.dot(x_tm1, x_tm1))

        # Compute residuals (eps) for GARCH fit
        eps = returns.iloc[1:] - phi * returns.shift(1).dropna()
        eps.index = returns.index[1:]

        # Step 2: GARCH(1,1) with zero mean
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
            # Retry with Powell method if SLSQP constraints fail
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
        mu_1 = f.mean.iloc[-1, 0] 
        var1 = f.variance.iloc[-1,0] 

        orig_ts = self.returns
        dates = orig_ts.index
        resid = self.fitted_model.resid.dropna() / self.fitted_model.conditional_volatility
        cov_values = resid.values[:-1]
        rv_values = resid.values[1:]
        time_idx   = resid.index         # corresponds to residuals at t = 2,…,n
        time_vals  = time_idx[1:] 
          
        ts = TimeSeries(
            time= time_vals,
            covariate_name = self.series_name,
            covariate=cov_values,
            rv_name=self.series_name + "_lag1",
            rv= rv_values
        )

        last_date = dates[-1]
        next_date = pd.to_datetime(last_date) + pd.tseries.offsets.BDay(1)
        
        return Forecast(residual_ts= ts, forecast_date = next_date, forecast_mu=mu_1, forecast_sigma2=var1)
    
