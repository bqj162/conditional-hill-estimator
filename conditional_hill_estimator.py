import sys

from HillEstimator import HillEstimator
from Plots import Plots
from parser import parse_command_line_arguments

def main():
        user_input = parse_command_line_arguments(sys.argv)

        for time_series in user_input.time_series:
                hill_estimator = HillEstimator(time_series)
                hill_estimate = hill_estimator.estimate() # estimate includes x, k, gamma
                plots = Plots(time_series, hill_estimate)







if __name__ == "__main__":
        main()