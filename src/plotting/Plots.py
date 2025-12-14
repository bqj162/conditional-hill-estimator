import plotly.graph_objects as go
from typing import Sequence
import numpy as np
import pandas as pd
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
        ax.bar(self.quantile_fit.index,  self.quantile_fit['obs'], width=1.0, label='Log-returns')
        ax.plot(self.quantile_fit.index, self.quantile_fit['x_hat']    , label='Conditional'  , linewidth=0.7, linestyle='dashed', color='red')
        ax.plot(self.quantile_fit.index, self.quantile_fit['x_hat_unc'], label='Unconditional', linewidth=0.7, linestyle ='dashdot', color='green')      
        # ax.scatter(self.quantile_fit.index,  self.quantile_fit['x_hat'],     s = 0.25 ,label='Conditional')
        # ax.scatter(self.quantile_fit.index,  self.quantile_fit['x_hat_unc'], s = 0.25 ,label='Unconditional')
        ax.set_xlabel('Date')
        ax.set_ylabel('Log-returns')
        ax.set_title(f"{self.time_series.rv_name}, q = {self.q}")
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"Plots/Forecast_quantiles_q_{self.q}_{self.time_series.rv_name}.pdf")
        plt.show()

    def plot_fitted_violations(self):
        fig, ax = plt.subplots()
       
        ax.bar(self.quantile_fit.index, self.quantile_fit['obs'], width=1.0, label='Log-returns')
       
        cond_exceed = self.quantile_fit['obs'] > self.quantile_fit['x_hat']
        unc_exceed  = self.quantile_fit['obs'] > self.quantile_fit['x_hat_unc']

        ax.scatter(self.quantile_fit.index[cond_exceed], 
                    self.quantile_fit['x_hat'][cond_exceed],  
                    marker='o', color='red', s=20, label='Exceed Cond.')

        ax.scatter(self.quantile_fit.index[unc_exceed], 
                    self.quantile_fit['x_hat_unc'][unc_exceed], 
                    marker='^', color='green', s=20, label='Exceed Uncond.')
  
        ax.set_xlabel('Date')
        ax.set_ylabel('Log-returns')
        ax.set_title(f"{self.time_series.rv_name}, q = {self.q}")
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"Plots/Forecast_violations_q_{self.q}_{self.time_series.rv_name}.pdf")
        plt.show()


    def plot_bias_MSE(bias_MSE_df):
        k = bias_MSE_df['k']
        lw = 1
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.plot(k, bias_MSE_df['bias']  , label = 'Unconditional',linewidth=lw, linestyle='dashed', color='red')
        ax1.plot(k, bias_MSE_df['bias_t'], label = 'Conditional'  ,linewidth=lw, linestyle ='dashdot', color='green')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, label='zero')
        ax2.plot(k, bias_MSE_df['MSE']   , label = 'Unconditional',linewidth=lw, linestyle='dashed', color='red')
        ax2.plot(k, bias_MSE_df['MSE_t'] , label = 'Conditional'  ,linewidth=lw, linestyle ='dashdot', color='green')
        ax1.set_xlabel('k_n')
        ax2.set_xlabel('k_n')
        ax1.set_ylabel('Bias')
        ax2.set_ylabel('MSE')
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.show()


    def plot_gamma_sim(gamma_df):
        k = gamma_df['k']
        lw = 1
        fig, ax = plt.subplots()
        ax.plot(k, gamma_df['gamma']  , label = 'gamma',linewidth=lw, linestyle='dashed', color='red')
        ax.plot(k, gamma_df['gamma_x'], label = 'gamma_x'  ,linewidth=lw, linestyle ='dashdot', color='green')
        ax.plot(k, gamma_df['target'], color='black', linestyle='--', linewidth=1, label='target')
        ax.set_xlabel('k_n')
        ax.set_ylabel('gammas')
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_2x2_grid_param(results_df: pd.DataFrame,
                            burn_ins: Sequence[int],
                            col_param: str,
                            col_values: Sequence,
                            figsize=(12, 8),
                            linewidth=1):
        """
        results_df must contain columns:
        at least ['burn_in', col_param, 'k', 'bias', 'bias_t', 'MSE', 'MSE_t'].

        burn_ins: sequence with exactly two burn_in values (first -> solid, second -> dashed)
        col_param: parameter name that will define the columns (e.g. 'x_eval' or 'q')
        col_values: values for the two columns (first two used)
        """
        if len(burn_ins) < 2 or len(col_values) < 2:
            raise ValueError("Provide at least two burn_ins and two col_values.")

        b0, b1 = int(burn_ins[0]), int(burn_ins[1])
        c0, c1 = col_values[0], col_values[1]

        results_df = results_df.sort_values([col_param, 'burn_in', 'k'])

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharex='col')

        style_map = {
            ('uncond', b0): ('red',  '-',  f'Uncond, n={b0}'),
            ('uncond', b1): ('red',  '--', f'Uncond, n={b1}'),
            ('cond',   b0): ('green','-',  f'Cond,   n={b0}'),
            ('cond',   b1): ('green','-.', f'Cond,   n={b1}'),
        }

        def plot_subplot(ax, col_val, metric):
            for (which, b), (color, ls, label) in style_map.items():
                sel = results_df[(results_df[col_param] == col_val) & (results_df['burn_in'] == b)]
                if sel.empty:
                    continue
                k = sel['k'].values
                k_over_n = k / b
                y = sel[metric].values if which == 'uncond' else sel[f"{metric}_t"].values
                ax.plot(k_over_n, y, label=label, color=color, linestyle=ls,linewidth=linewidth)

            if metric == 'bias':
                ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
                ax.set_ylabel('Bias')
            else:
                ax.set_ylabel('MSE')
            ax.set_xlabel('k/n')
            ax.set_title(f"{metric}, {col_param} = {col_val}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize='small')

        plot_subplot(axes[0,0], c0, 'bias')
        plot_subplot(axes[0,1], c1, 'bias')
        plot_subplot(axes[1,0], c0, 'MSE')
        plot_subplot(axes[1,1], c1, 'MSE')

        plt.tight_layout()
        plt.savefig(f"Plots/Bias_MSE_sim_{col_param}_eq_{col_values}_gamma.pdf")
        plt.show()

