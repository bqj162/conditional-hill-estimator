import numpy as np
import pandas as pd
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
        # kde = FFTKDE(bw='ISJ').fit(x_ranked)
        # h =  kde.bw**2 # /2 
        h = np.sqrt(k_n/n_x)
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

        return gamma
    
    def gamma_fixed_k_n_x_vectorized(X, Y, k_array, x):
        X = np.asarray(X)
        Y = np.asarray(Y)
        k_arr = np.asarray(k_array, dtype=int)
        n_x = X.size
        if Y.size != n_x:
            raise ValueError("X and Y must have same length")
        if np.any(k_arr < 1) or np.any(k_arr > n_x - 1):
            raise ValueError("k must satisfy 1 <= k <= n_x-1")
        
        m = k_arr.size
        X_ranked = rankdata(X) / (n_x + 1.0) 
        # h = np.sqrt(k_arr/n_x).astype(float) #shape (m,)
        h = np.sqrt(np.log(k_arr)/n_x).astype(float) #shape (m,)

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
    

    def unconditional_hill_estimator(Y , k_n):
        Y_sorted = np.sort(Y)
        n = len(Y_sorted)
    
        threshold = Y_sorted[n - k_n - 1]
        tail = Y_sorted[n - k_n : ]
        gamma = np.mean(np.log(tail / threshold))

        return gamma
    
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

    Hill = HillEstimator(time_series=Y_sim)
    # values = HillEstimator.gamma_fixed_k_n(X = X_unif, Y = Y_sim, k_n = k)
    est = Hill.unconditional_hill_estimator(k_n = int(k))
    print(est)






    # def batch_gamma_fixed_k_n_x(X_mat: np.ndarray, Y_mat: np.ndarray,
    #                         k_array: np.ndarray, x_arr: np.ndarray) -> np.ndarray:
    #     """
    #     X_mat, Y_mat: shape (n_samples, n)
    #     k_array: shape (m,)
    #     x_arr: shape (n_samples,) - conditioning x per sample (scalar per sample)
    #     Returns: gamma_mat shape (n_samples, m)
    #     """
    #     X_mat = np.asarray(X_mat)
    #     Y_mat = np.asarray(Y_mat)
    #     k_arr = np.asarray(k_array, dtype=int)
    #     x_arr = np.asarray(x_arr, dtype=float)
    #     n_samples, n = X_mat.shape
    #     m = k_arr.size

    #     # 1) compute ranks (1..n) per row and scaled rankdata = rank/(n+1)
    #     order = np.argsort(X_mat, axis=1)                # (ns, n)
    #     ranks = np.empty_like(order)
    #     rows = np.arange(n_samples)[:, None]
    #     ranks[rows, order] = np.arange(n)[None, :]       # ranks 0..n-1
    #     X_ranked = (ranks + 1.0) / (n + 1.0)             # (ns, n) matching rankdata

    #     # 2) sort Y and align X_ranked by Y-sorting
    #     sort_idx = np.argsort(Y_mat, axis=1)
    #     Y_sorted = np.take_along_axis(Y_mat, sort_idx, axis=1)         # (ns, n)
    #     Xs_sorted = np.take_along_axis(X_ranked, sort_idx, axis=1)    # (ns, n)

    #     # 3) compute x_eval per sample
    #     rank_of_x = np.sum(X_mat <= x_arr[:, None], axis=1) + 1        # (ns,)
    #     x_eval = rank_of_x / (n + 1.0)                                # (ns,)

    #     # 4) kernel matrix K_X of shape (ns, m, n)
    #     # diffs is (ns, n) ; expand to (ns, 1, n)
    #     diffs = x_eval[:, None, None] - Xs_sorted[:, None, :]         # (ns, 1, n)
    #     h = np.sqrt(k_arr / n)                                        # (m,)
    #     # broadcast: diffs (ns,1,n) and h (1,m,1) -> result (ns,m,n)
    #     K_X = norm.pdf(diffs, scale=h[None, :, None])                # (ns, m, n)

    #     # 5) normalize per (sample,k) row-sums along last axis
    #     row_sums = K_X.sum(axis=2, keepdims=True)                    # (ns, m, 1)
    #     # avoid divide-by-zero
    #     eps = np.finfo(float).eps
    #     row_sums[row_sums == 0] = eps
    #     W = K_X / row_sums                                            # (ns, m, n)

    #     # 6) s = 1 - cumsum(W) along last axis
    #     s = 1.0 - np.cumsum(W, axis=2)                                # (ns, m, n)

    #     # 7) find idx per (sample,k): first j where s < k/n
    #     thresh = k_arr / n                                            # (m,)
    #     mask = s < thresh[None, :, None]                              # (ns, m, n)
    #     any_true = mask.any(axis=2)                                   # (ns, m)
    #     first_true = np.argmax(mask, axis=2)                          # (ns, m), returns 0 if no True
    #     idx = np.where(any_true, first_true, n - 1)                   # (ns, m) fallback to last index

    #     # 8) q_n per (sample,k)
    #     row_idx = np.arange(n_samples)[:, None]
    #     q_n = Y_sorted[row_idx, idx]                                  # (ns, m)

    #     # 9) tail mask and log ratios
    #     j = np.arange(n)                                               # (n,)
    #     tail_mask = (j[None, None, :] >= idx[:, :, None])             # (ns, m, n)

    #     # log ratios: broadcast Y_sorted (ns,1,n) / q_n (ns,m,1)
    #     log_ratios = np.log(Y_sorted[:, None, :] / q_n[:, :, None])   # (ns, m, n)

    #     # 10) compute weighted sum over tail: sum_j W * tail_mask * log_ratios
    #     terms = W * tail_mask * log_ratios                            # (ns, m, n)
    #     numer = terms.sum(axis=2)                                      # (ns, m)

    #     gamma_mat = (n / k_arr[None, :]) * numer                       # (ns, m)
    #     return gamma_mat



    # def batch_unconditional_hill(Y_mat: np.ndarray, k_array: np.ndarray) -> np.ndarray:
    #     """
    #     Y_mat: shape (n_samples, n)  (each row is one sample's Y values)
    #     k_array: shape (m,)
    #     returns: hill matrix shape (n_samples, m)
    #     """
    #     Y_mat = np.asarray(Y_mat)
    #     k_arr = np.asarray(k_array, dtype=int)
    #     n_samples, n = Y_mat.shape
    #     m = k_arr.size

    #     # sort each row ascending and take logs
    #     Y_sorted = np.sort(Y_mat, axis=1)                 # (ns, n)
    #     logs = np.log(Y_sorted)                           # (ns, n)

    #     # sum of top-k logs = cumsum of descending logs
    #     top_logs_cumsum = np.cumsum(logs[:, ::-1], axis=1)  # (ns, n), index 0 is largest log
    #     # pick S_k for each k
    #     S_k = top_logs_cumsum[:, k_arr - 1]               # (ns, m)

    #     # log threshold log_u_k for each k
    #     cols = (n - k_arr - 1)                            # (m,)
    #     # gather using advanced indexing
    #     row_idx = np.arange(n_samples)[:, None]
    #     log_u_k = logs[row_idx, cols[None, :]]           # (ns, m)

    #     hill = (S_k / k_arr[None, :]) - log_u_k          # (ns, m)
    #     return hill