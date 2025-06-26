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
import plotly as plt

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
        q_series = quantileSeries(q = 0.95, 
                                  time_series = neg_log_returns,
                                  fitting_window= 500)

        fit = q_series.estimate()
        print(fit)
        
        # main()

