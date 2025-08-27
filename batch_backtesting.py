def run_batch_backtests(args_list, q_s, fitting_window):
    import pandas as pd
    from parser import parse_command_line_arguments
    from quantile_estimator import quantileSeries
    from backtesting import back_test

    results = []
    for args in args_list:
        user_input = parse_command_line_arguments(args, split=False)
        log_returns = user_input.time_series
        q_series = quantileSeries(q=q_s, time_series=log_returns, fitting_window=fitting_window)
        fit = q_series.estimate()
        test = back_test(fit, q_s, log_returns.rv_name)
        results.append(test)
    out = pd.concat(results, ignore_index=True)
    return out