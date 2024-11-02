import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Plots:
    def __init__(self, time_series, hill_estimate):
        self.time_series = time_series
        self.hill_estimate = hill_estimate

    def plot_2d(self):
        dates = self.time_series.time
        x = self.time_series.covariate
        y = self.time_series.rv
        #positive_indicies = [y > 0]

        ts_plot = make_subplots(rows=2, cols=1, subplot_titles=("covariate process", "rv process"))
        ts_plot.add_trace(go.Scatter(x = dates, y = x), row=1, col=1)
        ts_plot.add_trace(go.Scatter(x = dates, y = y), row=2, col=1)
        ts_plot.show()
        return ts_plot

    def plot_3d(self):
        fig = go.Figure(data=[go.Surface(z = self.hill_estimate['gamma_k_x'], x = self.hill_estimate['X'], y = self.hill_estimate['K'])])
        fig.update_layout(title='gamma_k_x', autosize=True)
        fig.show()
        return fig


