import sys
from parser import parse_command_line_arguments

def main():
        user_input = parse_command_line_arguments(sys.argv)
        #generate_plots(user_input)

if __name__ == "__main__":
        main()