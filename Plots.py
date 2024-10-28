import plotly.graph_objects as go
from plotly.subplots import make_subplots
from HillEstimator import HillEstimator
class Plots:
    def __init__(self, time_series, hill_estimate):
        self.time_series = time_series
        self.hill_estimate = hill_estimate

    def plot_2d(self):
        dates = self.user_input.get_time()
        x = self.user_input.get_covariate_process()
        y = self.user_input.get_rv_process()
        #positive_indicies = [y > 0]

        ts_plot = make_subplots(rows=2, cols=1, subplot_titles=("covariate process", "rv process"))
        ts_plot.add_trace(go.Scatter(x = dates, y = x), row=1, col=1)
        ts_plot.add_trace(go.Scatter(x = dates, y = y), row=2, col=1)
        ts_plot.show()
        return ts_plot

    def plot_3d(self):
        fig = go.Figure(data=[go.Surface(z = self.gamma, x = self.x, y = self.k)])
        fig.update_layout(title='gamma_k_x', autosize=True)
        return fig
        #fig.show()

