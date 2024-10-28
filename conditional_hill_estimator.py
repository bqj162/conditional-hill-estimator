import sys

from HillEstimator import HillEstimator
from Plots import Plots
from parser import parse_command_line_arguments

def main():
        user_input = parse_command_line_arguments(sys.argv)
        for time_series_transformed in user_input.time_series_transformed:
                hill_estimator = HillEstimator(time_series_transformed)
                hill_estimate = hill_estimator.estimate() # estimate includes x, k, gamma
                hill_plots = Plots(time_series_transformed, hill_estimate)






        plot = HillPlot(user_input = user_input)
        plot.plot_2d()
        plot.plot_3d()
        # hill_plots(estimator)





if __name__ == "__main__":
        main()