import pandas as pd

from TimeSeries import TimeSeries
from UserInput import UserInput

def parse_command_line_arguments(argv : list[str]):
    if len(argv) != 3:
        raise Exception("Wrong usage: conditional_hill_estimator.py [time_series_file] [kernel] [...]")

    time_series = parse_time_series_file(argv[1])

    return UserInput(time_series)


def parse_time_series_file(filename : str):
    time_series_df = pd.read_excel(io=filename)
    time_series = TimeSeries(time=time_series_df['time'], covariate=time_series_df['covariate'], rv=time_series_df['rv'])
    return time_series


def parse_stock_tickers():
    pass