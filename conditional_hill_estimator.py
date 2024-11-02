import sys
from HTML import HTML
from HillEstimator import HillEstimator
from Plots import Plots
from parser import parse_command_line_arguments

def main():
        user_input = parse_command_line_arguments(sys.argv)
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
        main()