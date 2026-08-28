import sys
from src.reporting.HTML import HTML
from src.estimators.HillEstimator import HillEstimator
from src.plotting.Plots import Plots
from src.data.parser import parse_command_line_arguments


def plot_marginals_and_3d_hill_plots_as_html():
    user_input = parse_command_line_arguments(sys.argv, split=True)
    plot_list = []
    upper_k_frac = 0.5
    for time_series in user_input.time_series:
        hill_estimator = HillEstimator(time_series)
        hill_estimate = hill_estimator.estimate(
            upper_k_frac=upper_k_frac
        )  # estimate includes x, k, gamma
        plots = Plots(time_series, hill_estimate)
        plot_list.append(plots.plot_2d_marginal())
        plot_list.append(plots.plot_3d())
    html = HTML(plot_list)
    html.generate_HTML()
