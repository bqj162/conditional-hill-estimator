import numpy as np
from scipy.stats import norm, rankdata
from scipy.stats import gaussian_kde
from KDEpy import FFTKDE


class HillEstimator:
    def __init__(self, time_series, grid_resolution = 200):
        self.time_series = time_series
        self.grid_resolution = grid_resolution
        self.hill_estimate = None

    def estimate(self):
        self.hill_estimate = self.gamma_full(X =  self.time_series.covariate, Y = self.time_series.rv)
        return self.hill_estimate


    def gamma_full(self, X, Y):
        x_vals = np.linspace(0.00, 1, num = self.grid_resolution)
        results = np.full((self.grid_resolution, self.grid_resolution), np.nan)

        for i in range(self.grid_resolution):
            fit = self.gamma_automatic(x_vals[i], X, Y)
            results[i,] = fit['gammas']

        return {'X':x_vals, 'K': fit['ks'], 'gamma_k_x': results}

    def gamma_automatic(self, x, X, Y):

        n_x = len(X)
        x_ranked = rankdata(X) / (n_x + 1)

        #kde = gaussian_kde(x_ranked, bw_method='scott')
        kde = FFTKDE(bw='ISJ').fit(x_ranked)
        h = kde.bw / 2
        w = norm.pdf(x - x_ranked, scale=np.sqrt(h))

        sort_idx = np.argsort(Y)
        y_sorted = Y[sort_idx]
        w_sorted = w[sort_idx] / np.sum(w)

        s = 1 - np.cumsum(w_sorted)

        ks = np.floor(np.linspace(0.01 * n_x, 0.5 * n_x, self.grid_resolution)).astype(int)
        gammas = np.full(n_x, np.nan)


        for i in range(self.grid_resolution):
            k = ks[i]
            idx = np.min(np.where(s < k / n_x))
            qn = y_sorted[idx]

            gammas[i] = (n_x / k) * np.sum(w_sorted[idx:] * np.log((y_sorted[idx:] / qn).astype(float)))

        # Return results
        return {'ks': ks, 'gammas': gammas[~np.isnan(gammas)]}
