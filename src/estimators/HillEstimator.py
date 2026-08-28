import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from KDEpy import FFTKDE


class HillEstimator:
    def __init__(self, time_series, grid_resolution: int = 200):
        self.time_series = time_series
        self.grid_resolution = grid_resolution
        self.hill_estimate = None

    def _k_grid(self, upper_k_frac: float, n: int) -> np.ndarray:
        if self.grid_resolution < 1:
            raise ValueError("grid_resolution must be positive")
        if not 0 < upper_k_frac < 1:
            raise ValueError(
                "upper_k_frac must lie strictly between 0 and 1"
            )

        lower_k = int(np.floor(0.01 * n))
        upper_k = int(np.floor(upper_k_frac * n))
        if lower_k < 2 or upper_k < 2:
            raise ValueError("Every k must be at least 2")
        if upper_k < lower_k:
            raise ValueError("upper_k_frac must define a nonempty k grid")

        ks = np.floor(
            np.linspace(lower_k, upper_k, self.grid_resolution)
        ).astype(int)

        if np.any(ks < 1) or np.any(ks >= n):
            raise ValueError("Every k must satisfy 1 <= k < sample size")

        return ks

    def estimate(self, upper_k_frac: float):
        if (self.time_series.covariate is not None):
            if len(self.time_series.covariate) < self.grid_resolution:
                raise Exception ('Covariate sequence is shorter that grid resolution')

        self.hill_estimate = self.gamma_full(X =  self.time_series.covariate,
                                             Y = self.time_series.rv,
                                             upper_k_frac=upper_k_frac)
        return self.hill_estimate


    def gamma_full(self, X, Y, upper_k_frac: float):
        x_vals = np.linspace(0.00, 1, num = self.grid_resolution)
        ks = self._k_grid(n = len(X), upper_k_frac = upper_k_frac)
        results = np.full((self.grid_resolution, self.grid_resolution), np.nan)

        for i, x in enumerate(x_vals):
            fit = self.gamma_automatic(x, X, Y, ks)
            results[i] = fit["gammas"]


        return {'X':x_vals, 'K': ks, 'gamma_k_x': results}

    # def gamma_automatic(self, x, X, Y, ks):
    #     n_x = len(X)
    #     x_ranked = rankdata(X) / (n_x + 1)
    #     #kde = gaussian_kde(x_ranked, bw_method='scott')
    #     kde = FFTKDE(bw='ISJ').fit(x_ranked)  # pyright: ignore[reportArgumentType]
    #     h = float(np.asarray(kde.bw).squeeze()) / 2

    #     sort_idx = np.argsort(Y)
    #     y_sorted = Y[sort_idx]
    #     x_sorted = x_ranked[sort_idx]
    #     w = norm.pdf(x - x_sorted, scale=np.sqrt(h))
    #     w_sorted = w / np.sum(w)
    #     s = 1 - np.cumsum(w_sorted)

    #     # ks = np.floor(np.linspace(0.01 * n_x, 0.5 * n_x, self.grid_resolution)).astype(int)
    #     gammas = np.full(self.grid_resolution, np.nan)

    #     for i in range(self.grid_resolution):
    #         k = ks[i]
    #         idx = np.min(np.where(s < k / n_x))
    #         qn = y_sorted[idx]
    #         gammas[i] = (n_x / k) * np.sum(w_sorted[idx:] * np.log((y_sorted[idx:] / qn).astype(float)))

    #     # Return results
    #     return {'ks': ks, 'gammas': gammas[~np.isnan(gammas)]}

    def gamma_automatic(self, x, X, Y, ks):
        n_x = len(X)
        x_ranked = rankdata(X) / (n_x + 1)
        #kde = gaussian_kde(x_ranked, bw_method='scott')
        kde = FFTKDE(bw='ISJ').fit(x_ranked)  # pyright: ignore[reportArgumentType]
        h = float(np.asarray(kde.bw).squeeze()) / 2

        sort_idx = np.argsort(Y)
        y_sorted = Y[sort_idx]
        x_sorted = x_ranked[sort_idx]
        w = norm.pdf(x - x_sorted, scale=np.sqrt(h))
        w_sorted = w / np.sum(w)
        s = 1 - np.cumsum(w_sorted)

        gammas = np.empty(len(ks), dtype=float)

        for i, k in enumerate(ks):
            candidates = np.flatnonzero(s < k / len(X))
            if candidates.size == 0:
                raise FloatingPointError(
                    f"Could not locate threshold for x={x}, k={k}"
                )

            idx = int(candidates[0])
            qn = y_sorted[idx]

            gammas[i] = (
                len(X) / k
                * np.sum(
                    w_sorted[idx:]
                    * np.log(y_sorted[idx:] / qn)
                )
            )

        invalid = ~np.isfinite(gammas)
        if np.any(invalid):
            raise FloatingPointError(
                f"Non-finite estimates for x={x}, "
                f"k={ks[invalid].tolist()}"
            )

        return {"ks": ks, "gammas": gammas}


    @staticmethod
    def gamma_fixed_k_n_x(X, Y, k_n, x):
        n_x = len(X)
        x_ranked = rankdata(X) / (n_x + 1)
        # kde = FFTKDE(bw='ISJ').fit(x_ranked)
        # h =  kde.bw**2 # /2
        # h = np.sqrt(k_n/n_x)
        h = np.sqrt(np.log(k_n)/n_x).astype(float)
        sort_idx = np.argsort(Y)
        y_sorted = Y[sort_idx]
        x_sorted = x_ranked[sort_idx]

        rank_of_x = np.sum(X <= x) + 1
        x_eval = rank_of_x / (n_x + 1)

        # K_X = norm.pdf(x_eval - x_sorted, scale = np.sqrt(h))
        K_X = norm.pdf(x_eval - x_sorted, scale = h)
        W = K_X/sum(K_X)
        s = 1 - np.cumsum(W)
        idx = np.min(np.where(s < k_n / n_x))
        q_n = y_sorted[idx]

        gamma = (n_x / k_n) * np.sum(W[idx:] * np.log((y_sorted[idx:] / q_n).astype(float)))

        return gamma, q_n

    @staticmethod
    def gamma_fixed_k_n_x_vectorized(X, Y, k_array, x):
        X = np.asarray(X)
        Y = np.asarray(Y)
        k_arr = np.asarray(k_array, dtype=int)
        n_x = X.size
        if Y.size != n_x:
            raise ValueError("X and Y must have same length")
        if np.any(k_arr < 1) or np.any(k_arr > n_x - 1):
            raise ValueError("k must satisfy 1 <= k <= n_x-1")

        X_ranked = rankdata(X) / (n_x + 1.0)
        # h = np.sqrt(k_arr/n_x).astype(float) #shape (m,)
        h = np.sqrt(np.log(k_arr)/n_x).astype(float) #shape (m,)
        # kde = FFTKDE(bw='ISJ').fit(X_ranked)
        # h = np.repeat(kde.bw,len(k_arr)) # /2

        sort_idx = np.argsort(Y)
        Y_sorted = Y[sort_idx]          # shape (n_x,)
        X_sorted = X_ranked[sort_idx]   # shape (n_x,)

        rank_of_x = np.sum(X <= x) + 1
        x_eval = rank_of_x / (n_x + 1.0)

        # compute kernel matrix K_X: shape (m, n_x)
        diffs = x_eval - X_sorted        # shape (n_x,)

        # expand for broadcasting: diffs[None, :] shape (1, n_x), h[:, None] shape (m, 1)
        K_X = norm.pdf(diffs[None, :], scale=h[:, None])   # shape (m, n_x)

        # sun weights along each X row
        row_sums = K_X.sum(axis=1, keepdims=True)         # shape (m, 1)

        # avoid division by zero
        row_sums[row_sums == 0] = np.finfo(float).eps
        W = K_X / row_sums                                # shape (m, n_x)

        # s = 1 - cumsum(W) each X row
        s = 1.0 - np.cumsum(W, axis=1)                    # shape (m, n_x)

        # threshold per row: threshold_i = k_arr[i] / n_x -> shape (m,)
        thresh = k_arr / n_x

        # mask where s < thresh (broadcast thresh[:,None] shape (m, 1))
        mask = s < thresh[:, None]                        # shape (m, n_x), boolean

        # For each row i we need the first index j where mask[i,j] is True.
        # If no True exists in a row set idx to n_x-1 (fallback).
        any_true = mask.any(axis=1)                       # shape (m,)
        # first_true: argmax gives first True if any True; otherwise returns 0
        first_true = np.argmax(mask, axis=1)              # shape (m,)
        # apply fallback (n_x-1) where no True
        idx = np.where(any_true, first_true, n_x - 1)     # shape (m,), dtype=int

        # get q_n per row
        q_n = Y_sorted[idx]                               # shape (m,)

        # Build a mask for tail sums: j indices
        j = np.arange(n_x)                                # shape (n_x,)
        tail_mask = j[None, :] >= idx[:, None]            # shape (m, n_x), boolean

        # compute log ratios matrix
        log_ratios = np.log(Y_sorted[None, :] / q_n[:, None])   # shape (m, n_x)

        # zero out elements before idx using tail_mask, then multiply by W and sum along X rows
        terms = W * tail_mask * log_ratios                # shape (m, n_x)
        numer = terms.sum(axis=1)                         # shape (m,)
        gamma_vec = (n_x / k_arr) * numer                 # shape (m,)

        return pd.DataFrame ({'gamma':  gamma_vec, 'q_n': q_n})


    @staticmethod
    def unconditional_hill_estimator(Y, k_n):
        Y_sorted = np.sort(Y)
        n = len(Y_sorted)

        threshold = Y_sorted[n - k_n - 1]
        tail = Y_sorted[n - k_n : ]
        gamma = np.mean(np.log(tail / threshold))

        return gamma

    @staticmethod
    def unconditional_hill_vectorized(Y, k_array):
        """
        Vectorized Hill estimator.
        Y : 1D array-like of positive observations (length n)
        k_array : array-like of integers, 1 <= k <= n-1
        returns : numpy array of gamma estimates (same shape as k_array)
        """
        Y     = np.asarray(Y)
        k_arr = np.asarray(k_array, dtype=int)
        n     = Y.size
        #Raise Value error if k_array is not "scaling"
        if np.any(k_arr < 1) or np.any(k_arr > n-1):
            raise ValueError("All k must satisfy 1 <= k <= n-1")

        Y_sorted = np.sort(Y)          # ascending order
        logs     = np.log(Y_sorted)    # log order statistics
        top_logs_cumsum = np.cumsum(logs[::-1])   # cumsum of descending logs, [::-1] reverses the order
        S_k       = top_logs_cumsum[k_arr - 1]    # sum of top k logs for each k
        log_u_k   = logs[n - k_arr - 1]           # log of threshold y_(n-k)
        gamma_vec = (S_k / k_arr) - log_u_k
        return gamma_vec


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

    est = HillEstimator.unconditional_hill_estimator(Y=Y_sim, k_n=int(k))
    print(est)
