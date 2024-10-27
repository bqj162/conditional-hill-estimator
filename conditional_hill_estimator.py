import sys
from Plot import Plot
from parser import parse_command_line_arguments

def main():
        user_input = parse_command_line_arguments(sys.argv)
        plot = Plot(user_input = user_input)
        plot.plot_2d()
        plot.plot_3d()
        # hill_plots(estimator)
        y = 1
if __name__ == "__main__":
        main()