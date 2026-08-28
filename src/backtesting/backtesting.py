from scipy.stats import binomtest
import pandas as pd
from src.data.TimeSeries import TimeSeries
from src.estimators.quantile_estimator import QuantileSeries

def back_test(fit, qs, name):
    results = []
    for q in qs:
        fit_q       = fit[fit["q"] == q]
        fit_pos     = fit_q[fit_q['obs'] > 0]
        test_length = len(fit_pos['obs'])
        expected_exceedances = int((1-q) * test_length)
        num_exceedances     = (fit_pos['x_hat']     < fit_pos['obs']).sum()
        num_exceedances_unc = (fit_pos['x_hat_unc'] < fit_pos['obs']).sum()
        b_test     = binomtest(num_exceedances    , n=test_length, p= 1 - q, alternative='two-sided')
        b_test_unc = binomtest(num_exceedances_unc, n=test_length, p= 1 - q, alternative='two-sided')
        frame = pd.DataFrame({
            "name": [name],"len": [test_length],"q": [q],
            "Expected": [expected_exceedances],
            "Conditional": [num_exceedances]  ,    "p_Conditional": [b_test.pvalue],
            "Unconditional": [num_exceedances_unc],"p_Unconditional": [b_test_unc.pvalue]
        })
        results.append(frame)
    out = pd.concat(results, ignore_index=True)
    return out


def run_single_backtest(
    time_series: TimeSeries,
    quantiles: list[float],
    fitting_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimator = QuantileSeries(
        time_series=time_series,
        q=quantiles,
        fitting_window=fitting_window,
    )
    forecasts = estimator.estimate()
    results = back_test(forecasts, quantiles, time_series.rv_name)
    return forecasts, results
