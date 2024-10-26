import sys
from Plot import Plot
from HillEstimator import HillEstimator
from parser import parse_command_line_arguments

def main():
        user_input = parse_command_line_arguments(sys.argv)
        hill_estimator = HillEstimator(user_input = user_input)
        hill_estimator.estimate()
        plot = Plot(hill_estimate_data = hill_estimator.hill_estimate)
        plot.plot_2d()
        plot.plot_3d()
        # hill_plots(estimator)
        y = 1
if __name__ == "__main__":
        main()