import pandas as pd
from UserInput import UserInput

def parse_command_line_arguments(argv : list[str]):
    if len(argv) != 3:
        raise Exception("Wrong usage: conditional_hill_estimator.py [time_series_file] [kernel] [...]")

    time_series = parse_time_series_file(argv[1])
    kernel = parse_kernel(argv[2])

    return UserInput(time_series, kernel)


def parse_time_series_file(filename : str):
    time_series = pd.read_excel(io=filename)
    return time_series


def parse_kernel(kernel : str):
    if kernel not in ["gaussian"]:
        raise Exception("Invalid kernel. Please choose from: gaussian, ...")
    return kernel