# Conditional Hill Estimation for Regularly Varying Markov Chains

[![checks](https://github.com/bqj162/conditional-hill-estimator/actions/workflows/checks.yml/badge.svg?branch=RV_markov_chains)](https://github.com/bqj162/conditional-hill-estimator/actions/workflows/checks.yml)

Research code for an ongoing paper on prediction of extremes in
regularly varying Markov chains. The repository implements conditional
tail-index estimation, financial time-series filtering, extreme-quantile
forecasting, coverage backtests, and reproducible parallel simulation studies.

This is an active research repository, not a production risk system or a
general-purpose Python package. The emphasis is on transparent statistical
experiments and reproducibility.

## Research question

The classical Hill estimator treats the upper tail index as constant. Here the
tail index may vary with a state or covariate:

$$
\widehat{\gamma}_{k_n}(x)
=
\frac{n}{k_n}
\frac{
\sum_{j=1}^{n}
K\!\left(\frac{x-X_j}{h_n}\right)
\log_+\!\left(\frac{Y_j}{q_n(x)}\right)
}{
\sum_{j=1}^{n}
K\!\left(\frac{x-X_j}{h_n}\right)
}.
$$

The implementation rank-transforms the covariate, applies Gaussian kernel
weights, estimates a local threshold and tail index, and compares the resulting
conditional forecast with an unconditional Hill benchmark.

## What is implemented

- Conditional and unconditional Hill estimators, including vectorised
  implementations used in Monte Carlo experiments.
- Rolling AR(1)-GARCH(1,1) filtering of financial log returns.
- One-step-ahead conditional and unconditional extreme-quantile forecasts.
- Binomial coverage tests for realised forecast exceedances.
- Pareto and Fréchet Markov-chain simulators with explicit random-number
  generators and deterministic seeding.
- Parameter-grid simulation studies parallelised with joblib.
- Matplotlib and Plotly reporting for estimates, forecasts, violations, bias,
  and mean squared error.

The financial forecasting pipeline converts log returns to losses, estimates
the AR-GARCH model on each rolling window, applies the tail estimators to
positive standardised residuals, and maps the forecast back to the return
scale.

## Repository layout

| Path | Purpose |
| --- | --- |
| **src/estimators/** | Conditional Hill, unconditional Hill, AR-GARCH, and rolling quantile forecasts |
| **src/simulation/** | Markov-chain simulators and parallel Monte Carlo studies |
| **src/backtesting/** | Exceedance counts, binomial tests, and batch backtests |
| **src/data/** | Market-data cache, CSV parsing, transforms, and tail preparation |
| **src/plotting/** | Forecast and simulation figures |
| **tests/** | Focused estimator, alignment, data-contract, and reproducibility checks |
| **Plots/forecasting/** | Selected rolling-forecast and violation figures |
| **Plots/simulation/** | Simulation figures grouped by innovation family |

## Setup

Python 3.12 is the currently tested version.

    git clone --branch RV_markov_chains https://github.com/bqj162/conditional-hill-estimator.git
    cd conditional-hill-estimator
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Market data are downloaded through yfinance and cached as Parquet files under
the user's cache directory. The cache is not part of the repository.

## Running the rolling financial forecast

The current entry point estimates the 0.95 loss quantile with a 500-observation
rolling fitting window and reports conditional and unconditional coverage
tests:

    python main.py \
      --stocks "^GDAXI" \
      --from_date "2005-01-01" \
      --to_date "2025-07-01" \
      --transform_type log_diff

For backwards compatibility, the repeated form **^GDAXI,^GDAXI** is also
accepted by the univariate forecast entry point.

CSV input must contain exactly three columns in this order: time, covariate,
and response.

    python main.py --file_path path/to/time_series.csv --transform_type log_diff

The rolling forecast is intentionally computationally expensive: an
AR-GARCH model and two tail estimators are fitted at each date.

## Checks

The branch runs the same checks locally and in GitHub Actions:

    python -m pip install -r requirements-dev.txt
    ruff check .
    pyright
    python -m pytest

The tests are deliberately focused on the contracts most important for the
research results: scalar/vectorised estimator agreement, one-step forecast
alignment, loss-tail construction, deterministic simulation, and input
validation.

## Selected outputs

- [Conditional and unconditional forecast violations for DAX losses](Plots/forecasting/GDAXI_loss_q0.95_violations.pdf)
- [Conditional and unconditional DAX loss-quantile forecasts](Plots/forecasting/GDAXI_loss_q0.95_quantiles.pdf)
- [Fréchet simulation bias and MSE across quantile levels](Plots/simulation/frechet/bias_mse_q_0.95_0.99.pdf)

These figures are experiment outputs rather than benchmark claims. The paper
and simulation design are still being developed. Newly generated forecast
filenames also include their forecast-date range, while simulation figures
record the innovation family in their title, filename, directory, and PDF
metadata.

## Current limitations

- Threshold and bandwidth choices are research parameters, not fully automatic
  tuning procedures.
- Coverage is currently assessed with an unconditional binomial exceedance
  test; independence and conditional-coverage diagnostics are planned.
- The command-line entry point exposes the main financial forecast but not
  every simulation parameter.
- yfinance is convenient for reproducible examples but is not a production
  market-data source.

## Reference

Hill, B. M. (1975). A Simple General Approach to Inference About the Tail of a
Distribution. *The Annals of Statistics*, 3(5), 1163–1174.

## License

MIT.
