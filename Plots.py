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

        ts_plot = make_subplots(rows=2, cols=1, subplot_titles=("Covariate process: " + self.time_series.covariate_name, "Regularly varying process: " + self.time_series.rv_name))
        ts_plot.add_trace(go.Scatter(x = dates, y = x, showlegend=False), row=1, col=1)
        ts_plot.add_trace(go.Scatter(x = dates, y = y, showlegend=False), row=2, col=1)
        return ts_plot

    def plot_3d(self):
        fig = go.Figure(data=[go.Surface(z = self.hill_estimate['gamma_k_x'],  y = self.hill_estimate['X'], x = self.hill_estimate['K'])])
        title = "Gamma_k_n estimates for " + "Covariate process: " + self.time_series.covariate_name + ", Regularly varying process: " + self.time_series.rv_name
        fig.update_layout(title=title, autosize=True)
        fig.update_scenes(xaxis_title_text='k_n',
                          yaxis_title_text='x',
                          zaxis_title_text='gamma_k_x')
        return fig
