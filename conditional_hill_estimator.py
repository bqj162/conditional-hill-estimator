import sys
from HTML import HTML
from HillEstimator import HillEstimator
from Plots import Plots
from parser import parse_command_line_arguments
from ts_fitting import AR1GARCG11
import pandas as pd
from ResidualSeries import ResidualSeries
from TimeSeries import TimeSeries
from quantile_estimator import quantileSeries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy
import numpy as np


def main():
        user_input = parse_command_line_arguments(sys.argv, split=True)
        plot_list = []
        for time_series in user_input.time_series:
                hill_estimator = HillEstimator(time_series)
                hill_estimate = hill_estimator.estimate() # estimate includes x, k, gamma
                plots = Plots(time_series, hill_estimate)
                plot_list.append(plots.plot_2d())
                plot_list.append(plots.plot_3d())
        html = HTML(plot_list)
        html.generate_HTML()

if __name__ == "__main__":
        
        user_input = parse_command_line_arguments(sys.argv, split = False)
        neg_log_returns = user_input.time_series
        chosen_q = 0.95
        q_series = quantileSeries(q = chosen_q, 
                                  time_series = neg_log_returns,
                                  fitting_window= 1000)

        fit = q_series.estimate()
        print(fit) 

        fit_pos = fit[fit['obs'] > 0]
        test_length = len(fit_pos['obs'])
        expected_exceedances = int((1-chosen_q) * test_length)

        print(f"Expected exceedances: {expected_exceedances}")
        num_exceedances     = (fit_pos['x_hat']     < fit_pos['obs']).sum()
        print(f"Conditional exceedances: {num_exceedances}")
        num_exceedances_unc = (fit_pos['x_hat_unc'] < fit_pos['obs']).sum()
        print(f"Unconditional exceedances: {num_exceedances_unc}")    

        b_test     = scipy.stats.binomtest(num_exceedances    , n=test_length, p= 1 - chosen_q, alternative='two-sided')
        print(f"Conditional p‐value: {b_test.pvalue:.4f}")

        b_test_unc = scipy.stats.binomtest(num_exceedances_unc, n=test_length, p= 1 - chosen_q, alternative='two-sided')
        print(f"unconditional p‐value: {b_test_unc.pvalue:.4f}")

        fig, ax = plt.subplots()
        ax.bar(fit.index,   fit['obs'], width=1, label='obs')
        ax.scatter(fit.index,  fit['x_hat'],     s = 0.35 ,label='Conditional')
        ax.scatter(fit.index,  fit['x_hat_unc'], s = 0.35 ,label='Unconditional')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
        
        # main()

