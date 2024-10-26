import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from hill import gamma_automatic


class HillEstimator:
    def __init__(self, user_input, hill_estimate = None, grid_resolution = 200):
        self.user_input = user_input
        self.hill_estimate = hill_estimate
        self.grid_resolution = grid_resolution

    def estimate(self):
        X = self.user_input.get_covariate_process()
        Y = self.user_input.get_rv_process()
        self.hill_estimate = self.gamma_full(X = X, Y = Y)


    def gamma_full(self, X, Y):
        x_vals = np.linspace(0.03, 0.97, num = self.grid_resolution)
        results = np.full((len(x_vals), 100), np.nan)

        for i in range(len(x_vals)):
            fit = gamma_automatic(x_vals[i], X, Y)
            gammas = fit['gammas']
            results[i, :len(gammas)] = gammas[:100]

        return pd.DataFrame(data= {'X':x_vals, 'K': fit['ks'], 'gamma_k_x': results})

    def gamma_automatic(self, x, X, Y):

        n_x = len(X)
        x_ranked = rankdata(X) / (n_x + 1)

        kde = gaussian_kde(x_ranked, bw_method='scott')
        h = kde.factor ** 2
        w = norm.pdf(x - x_ranked, scale=np.sqrt(h))

        sort_idx = np.argsort(Y)
        y_sorted = Y[sort_idx]
        w_sorted = w[sort_idx] / np.sum(w)

        s = 1 - np.cumsum(w_sorted)

        ks = np.floor(np.linspace(0.01 * n_x, 0.5 * n_x, self.grid_resolution)).astype(int)
        gammas = np.full(n_x, np.nan)
        qkn = np.full(n_x, np.nan)

        for k in ks:
            idx = np.min(np.where(s < k / n_x))
            qn = y_sorted[idx]
            qkn[k] = qn
            gammas[k] = (n_x / k) * np.sum(w_sorted[idx:] * np.log(y_sorted[idx:] / qn))

        # Return results
        return {'ks': ks, 'gammas': gammas[~np.isnan(gammas)]}
