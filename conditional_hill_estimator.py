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
        # from_date = "1995-01-01"
        # to_date = "2025-07-01"
        # args_list =[
        #         ["-s", "^GSPC,^GSPC"  ,"-fd", from_date, "-td", to_date,"-t", "log_diff","-l", "0"],
        #         ["-s", "^GDAXI,^GDAXI","-fd", from_date, "-td", to_date,"-t", "log_diff","-l", "0"],
        #         ["-s", "BMW.DE,BMW.DE","-fd", from_date, "-td", to_date,"-t", "log_diff","-l", "0"],
        #         ["-s", "GBP=X,GBP=X"  ,"-fd", from_date, "-td", to_date,"-t", "log_diff","-l", "0"],
        #         ["-s", "GC=F,GC=F"    ,"-fd", from_date, "-td", to_date,"-t", "log_diff","-l", "0"]   
        # ]
        
        # q_s = [0.95, 0.99, 0.995]
        # results = []
        # for args in args_list:
        #         user_input = parse_command_line_arguments(args, split = False)
        #         neg_log_returns = user_input.time_series
        #         q_series = quantileSeries(q = q_s, 
        #                                 time_series = neg_log_returns,
        #                                 fitting_window= 1500)
        #         fit = q_series.back_test()
        #         results.append(fit)
        # out = pd.concat(results, ignore_index=True)
        # print(out)
        # as_good_as  = (out['p_Unconditional']     <= out['p_Conditional']).sum()
        # print(f"Conditional as good as Unconditional: {as_good_as} out of 15")
        # better_than  = (out['p_Unconditional']     < out['p_Conditional']).sum()
        # print(f"Conditional better than Unconditional: {better_than} out of 15")
        # conditional_sig  = (0.05    <= out['p_Conditional']).sum()
        # print(f"Conditional significant: {conditional_sig} out of 15")
        # unconditional_sig  = (0.05    <= out['p_Conditional']).sum()
        # print(f"Unconditional significant: {unconditional_sig} out of 15")

        #-------------------One ts below-----------------------
        user_input = parse_command_line_arguments(sys.argv[1:], split=False)
        neg_log_returns = user_input.time_series
        chosen_q = [0.995]
        
        q_series = quantileSeries(q = chosen_q, 
                                  time_series = neg_log_returns,
                                  fitting_window= 1500)

        fit = q_series.estimate()
        print(fit) 

        fit_pos = fit[fit['obs'] > 0]
        test_length = len(fit_pos['obs'])
        expected_exceedances = int((1-chosen_q[0]) * test_length)

        print(f"Expected exceedances: {expected_exceedances}")
        num_exceedances     = (fit_pos['x_hat']     < fit_pos['obs']).sum()
        print(f"Conditional exceedances: {num_exceedances}")
        num_exceedances_unc = (fit_pos['x_hat_unc'] < fit_pos['obs']).sum()
        print(f"Unconditional exceedances: {num_exceedances_unc}")    

        b_test     = scipy.stats.binomtest(num_exceedances    , n=test_length, p= 1 - chosen_q[0], alternative='two-sided')
        print(f"Conditional p‐value: {b_test.pvalue:.4f}")

        b_test_unc = scipy.stats.binomtest(num_exceedances_unc, n=test_length, p= 1 - chosen_q[0], alternative='two-sided')
        print(f"unconditional p‐value: {b_test_unc.pvalue:.4f}")

        fig, ax = plt.subplots()
        ax.bar(fit.index,   fit['obs'], width=2.5, label='obs')
        # ax.plot(fit.index, fit['x_hat']    , label='Conditional'  , linewidth=0.5, linestyle='dashed', color='red')
        # ax.plot(fit.index, fit['x_hat_unc'], label='Unconditional', linewidth=0.5, linestyle ='dashdot', color='green')      
        ax.scatter(fit.index,  fit['x_hat'],     s = 0.25 ,label='Conditional')
        ax.scatter(fit.index,  fit['x_hat_unc'], s = 0.25 ,label='Unconditional')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"Forecast_quantiles_q_{chosen_q[0]}_{neg_log_returns.rv_name}.pdf")
        plt.show()
        

        

        # main()

