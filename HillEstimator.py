import numpy as np
from scipy.stats import norm, rankdata
from scipy.stats import gaussian_kde
from KDEpy import FFTKDE


class HillEstimator:
    def __init__(self, time_series, grid_resolution: int = 200):
        self.time_series = time_series
        self.grid_resolution = grid_resolution
        self.hill_estimate = None

    def estimate(self):
        if (self.time_series.covariate is not None):
            if len(self.time_series.covariate) < self.grid_resolution:
                raise Exception ('Covariate sequence is shorter that grid resolution')
            
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

        sort_idx = np.argsort(Y)
        y_sorted = Y[sort_idx]
        x_sorted = x_ranked[sort_idx]
        w = norm.pdf(x - x_sorted, scale=np.sqrt(h))
        w_sorted = w / np.sum(w)
        s = 1 - np.cumsum(w_sorted)

        ks = np.floor(np.linspace(0.01 * n_x, 0.5 * n_x, self.grid_resolution)).astype(int)
        gammas = np.full(self.grid_resolution, np.nan)

        for i in range(self.grid_resolution):
            k = ks[i]
            idx = np.min(np.where(s < k / n_x))
            qn = y_sorted[idx]
            gammas[i] = (n_x / k) * np.sum(w_sorted[idx:] * np.log((y_sorted[idx:] / qn).astype(float)))

        # Return results
        return {'ks': ks, 'gammas': gammas[~np.isnan(gammas)]}


    def gamma_fixed_k_n_x(X, Y, k_n, x):
        n_x = len(X)
        x_ranked = rankdata(X) / (n_x + 1)
        kde = FFTKDE(bw='ISJ').fit(x_ranked)
        h =  kde.bw**2 # /2
        sort_idx = np.argsort(Y)
        y_sorted = Y[sort_idx]
        x_sorted = x_ranked[sort_idx]

        rank_of_x = np.sum(X <= x) + 1  
        x_eval = rank_of_x / (n_x + 1)

        # K_X = norm.pdf(x_eval - x_sorted, scale = np.sqrt(h))
        K_X = norm.pdf(x_eval - x_sorted, scale = np.sqrt(k_n/n_x))
        W = K_X/sum(K_X)
        s = 1 - np.cumsum(W)
        idx = np.min(np.where(s < k_n / n_x))
        q_n = y_sorted[idx]
       
        gamma = (n_x / k_n) * np.sum(W[idx:] * np.log((y_sorted[idx:] / q_n).astype(float)))

        return gamma

    def unconditional_hill_estimator(Y , k_n):
        Y_sorted = np.sort(Y)
        n = len(Y_sorted)
    
        threshold = Y_sorted[n - k_n - 1]
        tail = Y_sorted[n - k_n : ]
        gamma = np.mean(np.log(tail / threshold))

        return gamma

if __name__ == "__main__":
    def gamma(s):
        return 0.3 + 0.2 * norm.pdf(s, loc = 0.2, scale = 0.05)
    
    rng = np.random.default_rng(1)
    n = 1000
    U_unif = rng.uniform(size = n)
    X_unif = rng.uniform(size = n)
    # Y_sim = (1-U_unif)**(-gamma(X_unif))
    Y_sim = (1-U_unif)**(-2)
    k = n / 10

    Hill = HillEstimator(time_series=Y_sim)
    # values = HillEstimator.gamma_fixed_k_n(X = X_unif, Y = Y_sim, k_n = k)
    est = Hill.unconditional_hill_estimator(k_n = int(k))
    print(est)