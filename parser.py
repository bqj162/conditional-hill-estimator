import pandas as pd
from TimeSeries import TimeSeries
from UserInput import UserInput
import argparse
valid_transform_types = [None, "log_diff"]
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stocks", dest="stock_tickers")
parser.add_argument("-t", "--transform_type", dest="transform_type")
parser.add_argument("-f", "--file_path", dest="time_series")
args = parser.parse_args()


def parse_command_line_arguments(argv : list[str]):
    if args.stock_tickers is not None and args.time_series is not None:
        raise Exception("Either provide stock tickers or your own time series, but not both.")

    stock_tickers = None
    time_series = None

    if args.stock_tickers is not None:
        stock_tickers = args.stock_tickers.split(",")
    elif args.time_series is not None:
        time_series = parse_time_series_file(args.time_series)
    else:
        raise Exception("Need to provide either stock tickers or your own time series.")

    transform_type = parse_transform_type(args.transform_type)
    user_input = UserInput(stock_tickers=stock_tickers, transform_type=transform_type, time_series=time_series)
    return user_input




def parse_transform_type(transform_type):
    if transform_type not in valid_transform_types:
        raise Exception(f"Not a valid transform_type, please use: {[transform_type for transform_type in valid_transform_types]}")
    return transform_type


def parse_time_series_file(filename):
    time_series_df = pd.read_excel(io=filename)
    time_series = TimeSeries(time=time_series_df['time'], covariate=time_series_df['covariate'], rv=time_series_df['rv'])
    return time_series


def convert_prices_to_time_series(prices):
    pass
