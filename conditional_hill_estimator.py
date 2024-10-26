import sys
from parser import parse_command_line_arguments

def main():
        user_input = parse_command_line_arguments(sys.argv)

        # estimator = hill_estimator(user_input)
        # hill_plots(estimator)

if __name__ == "__main__":
        main()