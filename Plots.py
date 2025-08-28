import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


class Plots:
    def __init__(self, time_series = None, hill_estimate = None, quantile_fit = None, q = None):
        self.time_series = time_series
        self.hill_estimate = hill_estimate
        self.quantile_fit = quantile_fit
        self.q = q  

    def plot_2d_marginal(self):
        dates = self.time_series.time
        x = self.time_series.covariate
        y = self.time_series.rv

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

    def plot_fit(self):
        fig, ax = plt.subplots()
        ax.bar(self.quantile_fit.index,   self.quantile_fit['obs'], width=2.5, label='obs')
        # ax.plot(fit.index, fit['x_hat']    , label='Conditional'  , linewidth=0.5, linestyle='dashed', color='red')
        # ax.plot(fit.index, fit['x_hat_unc'], label='Unconditional', linewidth=0.5, linestyle ='dashdot', color='green')      
        ax.scatter(self.quantile_fit.index,  self.quantile_fit['x_hat'],     s = 0.25 ,label='Conditional')
        ax.scatter(self.quantile_fit.index,  self.quantile_fit['x_hat_unc'], s = 0.25 ,label='Unconditional')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"Forecast_quantiles_q_{self.q}_{self.time_series.rv_name}.pdf")
        plt.show()

    def plot_bias_MSE(bias_MSE_df):
        k = bias_MSE_df['k']
        lw = 1
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.plot(k, bias_MSE_df['bias']  , label = 'Unconditional',linewidth=lw, linestyle='dashed', color='red')
        ax1.plot(k, bias_MSE_df['bias_t'], label = 'Conditional'  ,linewidth=lw, linestyle ='dashdot', color='green')
        ax2.plot(k, bias_MSE_df['MSE']   , label = 'Unconditional',linewidth=lw, linestyle='dashed', color='red')
        ax2.plot(k, bias_MSE_df['MSE_t'] , label = 'Conditional'  ,linewidth=lw, linestyle ='dashdot', color='green')
        plt.tight_layout()
        plt.show()