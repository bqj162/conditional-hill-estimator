import sys
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from HTML import HTML
from HillEstimator import HillEstimator
from Plots import Plots
from quantile_estimator import quantileSeries
from parser import parse_command_line_arguments
from backtesting import back_test
from batch_backtesting import run_batch_backtests

def main():
        user_input = parse_command_line_arguments(sys.argv, split=True)
        plot_list = []
        for time_series in user_input.time_series:
                hill_estimator = HillEstimator(time_series)
                hill_estimate = hill_estimator.estimate() # estimate includes x, k, gamma
                plots = Plots(time_series, hill_estimate)
                plot_list.append(plots.plot_2d_marginal())
                plot_list.append(plots.plot_3d())
        html = HTML(plot_list)
        html.generate_HTML()

if __name__ == "__main__":
        from_date = "2015-01-01"
        to_date = "2025-07-01"
        args_list =[
                ["-s", "^GSPC,^GSPC"  ,"-fd", from_date, "-td", to_date,"-t", "log_diff","-l", "0"],
                ["-s", "^GDAXI,^GDAXI","-fd", from_date, "-td", to_date,"-t", "log_diff","-l", "0"],
                ["-s", "BMW.DE,BMW.DE","-fd", from_date, "-td", to_date,"-t", "log_diff","-l", "0"],
                ["-s", "GBP=X,GBP=X"  ,"-fd", from_date, "-td", to_date,"-t", "log_diff","-l", "0"],
                ["-s", "GC=F,GC=F"    ,"-fd", from_date, "-td", to_date,"-t", "log_diff","-l", "0"]   
        ]
        
        q_s = [0.95, 0.99, 0.995]
        out = run_batch_backtests(args_list, q_s, fitting_window=400)
        
        print(out)
        as_good_as  = (out['p_Unconditional']     <= out['p_Conditional']).sum()
        print(f"Conditional as good as Unconditional: {as_good_as} out of 15")
        better_than  = (out['p_Unconditional']     < out['p_Conditional']).sum()
        print(f"Conditional better than Unconditional: {better_than} out of 15")
        conditional_sig  = (0.05    <= out['p_Conditional']).sum()
        print(f"Conditional significant: {conditional_sig} out of 15")
        unconditional_sig  = (0.05    <= out['p_Conditional']).sum()
        print(f"Unconditional significant: {unconditional_sig} out of 15")

        #-------------------One ts below-----------------------
        # user_input  = parse_command_line_arguments(sys.argv[1:], split=False)
        # log_returns = user_input.time_series
        # chosen_q    = [0.995]

        # q_series = quantileSeries(q = chosen_q, time_series = log_returns,fitting_window= 500)
        # fit      = q_series.estimate()
        # b_test   = back_test(fit, chosen_q, log_returns.rv_name)
        # plots    = Plots(time_series=log_returns, hill_estimate=None, quantile_fit=fit, q=chosen_q[0])
        # plots.plot_fit()

        # main()

