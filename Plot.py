import plotly.graph_objects as go
class Plot:
    def __init__(self, hill_estimate_data):
        self.hill_estimate_data = hill_estimate_data
        self.x     = (self.hill_estimate_data)['X']
        self.k     = (self.hill_estimate_data)['K']
        self.gamma = (self.hill_estimate_data)['gamma_k_x']
        self.three_d = self.plot_3d



    def plot_2d(self):
        pass

    def plot_3d(self):
        fig = go.Figure(data=[go.Surface(z=gamma, x=x, y=k)])
        fig.update_layout(title='gamma_k_x', autosize=True)
        return fig
        #fig.show()


