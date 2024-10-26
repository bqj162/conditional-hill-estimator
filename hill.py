import numpy as np
from scipy.stats import norm, rankdata
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

grid_resolution = 200
def gamma_automatic(x, X, Y):
    n_x = len(X)
    x_ranked = rankdata(X) / (n_x + 1)

    kde = gaussian_kde(x_ranked, bw_method='scott')
    h = kde.factor ** 2
    w = norm.pdf(x - x_ranked, scale=np.sqrt(h))

    sort_idx = np.argsort(Y)
    y_sorted = Y[sort_idx]
    w_sorted = w[sort_idx] / np.sum(w)

    s = 1 - np.cumsum(w_sorted)

    ks     = np.floor(np.linspace(0.01 * n_x, 0.5 * n_x, grid_resolution)).astype(int)
    gammas = np.full(n_x, np.nan)
    qkn    = np.full(n_x, np.nan)

    for k in ks:
        idx    = np.min(np.where(s < k / n_x))
        qn     = y_sorted[idx]
        qkn[k] = qn
        gammas[k] = (n_x / k) * np.sum(w_sorted[idx:] * np.log(y_sorted[idx:] / qn))

    # Return results
    return {'ks': ks, 'gammas': gammas[~np.isnan(gammas)]}

def gamma_full(X,Y):



x_vals = np.linspace(0.03, 0.97, num = 100)

results = np.full((len(x_vals), 100), np.nan)
bandwidth_output = np.full(len(x_vals), np.nan)

for i in range(len(x_vals)):
    fit = gamma_automatic(x_vals[i], X, Y)
    gammas = fit['gammas']

    results[i, :len(gammas)] = gammas[:100]
    bandwidth_output[i] = np.sqrt(fit['bandwidth'])

# Let's check the implementation trough simulation
np.random.seed(3)
n = 30000
# Define gamma function
def gamma(x):
    return 0.3 + 0.2 * norm.pdf(x, loc=0.2, scale=0.05)
X = np.random.uniform(size=n)
#arma_model = ArmaProcess([1, -0.2], [1])
#arma_samples = arma_model.generate_sample(nsample=n)
U = np.random.uniform(low=0.0, high=1.0, size=n)

Y = (1 - U) ** -gamma(X) #This is conditionally regularly varying
#Must finish SOON


#print(results[0,0], bandwidth_output[0])

x_for_plot, K = np.meshgrid(x_vals, fit['ks'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_for_plot, K, results.T, cmap='coolwarm', edgecolor='none')

# Add labels and title
ax.set_title("gamma_k(x)")
ax.set_xlabel('x')
ax.set_ylabel('k')
ax.set_zlabel('gamma(k,x)')

# Adjust viewing angles (similar to R's theta and phi)
ax.view_init(elev=15, azim=140)
print("hi")
plt.savefig('gamma_k_plot.png', dpi=300)  # You can change the filename and format as needed
# Close the plot (optional, to free up resources)
plt.close(fig)



