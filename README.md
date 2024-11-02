# Conditional Hill Estimator
#### Video Demo:  <URL HERE>
#### Description: 
The Hill estimator presented by Hill (1975) estimates the tail index for regularly varying sequence of random variables. Common goals of such estimator is to establish asymptotic properties such as consistency and asymptotic normality, which have been studied in the case of the Hill estimator for both iid, mixing sequences and networks. 
In this project, the task of analyzing extremal events for mixing regularly varying time series in the presence of random covariates is considered. 
The goal of this program is to implement a Nadaraya-Watson estimator of the conditional tail index of a stationary sequence $(Y_n)_{n \in \mathbb{Z}}$ given a covariate sequence $(X_n)_{n \in \mathbb{Z}}$.  
$$
    \widehat{\gamma}_{k_n}(x) =
   \frac{n}{k_{n}} \frac{\sum_{j=1}^{n}K\left(\frac{x-X_{j}}{h_{n}}\right)\log_{+} \left(\frac{Y_j}{q_n}\right)}
   {\sum_{j=1}^{n}K\left(\frac{x-X_{j}}{h_{n}}\right)},
$$
where $K$ is a kernel satisfying certain regularity conditions, $k_n$ is an intermediate sequence diverging to infinity and $q_n = F^{\leftarrow, x}_{n}\left(1-\frac{k_n}{n}\right)$ is the generalized inverse of the conditional distribution of $Y_0$ given $X_0 = x$. 

Input: The user of the program have two options regarding input. The first option is to upload a csv file with three columns, where the first column are dates, the second column is the value of time series of interest $(Y_n)_{n \in \mathbb{Z}}$ and the third column is the value of the covariate process of interest $(X_n)_{n \in \mathbb{Z}}$.
The second type of input to be provided is the kernel, the user wants to use in the above estimator. While asymptotic normality of $\widehat{\gamma}_{k_n}(x)$ is proved used kernel with bounded support, such as the uniform kernel or the epanechnikov kernel (who both have support on $[-1,1]$), the user have the freedom to pick a kernel with unbounded support such as the Gaussian kernel.

The second option is to allow the user to simulate from a distribution for a selected tail index function $x \mapsto \gamma(x)$.

Architecture of program: The input of the user is handled with several "get"-function from UserInput.py. The input is parsed onto the rest of the program with the "parser"-functions in parser.py. The latter contains the function "parse_command_line_arguments", which raises a meaningfull exception if the command line arguments are not specified correctly.

It is important to note that the above estimator needs to be implemented on a stationary, non-negative time series. As many stationary time series such as log-return are centered around zero, the estimation routines in this program is split into estimating the tail index of both of the following to set of time series $(Y_n,X_n)_{Y_n > 0}$ and $(Y_n,X_n)_{Y_n < 0}$. This i advantageous since it allows for the positive part of $Y_n$ given $X_n$ to have a different tail index than that of the negative side.

The file HillEstimator.py implement the above estimator in the function "estimate". It outputs a list of the vectors $x \in \mathbb{R}^{n}$ and $k_n \in \mathbb{R}^{n}$, as well as the array $(x,k_n) \mapsto \widehat{\gamma}_{k_n}(x) \in \in \mathbb{R}^{\otimes 2n}$, which is the estimator of interest.

The file Plot.py implements three kinds of plots. 1) Plotting the given time series side by side for visual comparison. 2) Plotting the conditional hill estimator either for fixed $x$ as a function of $k_n$ or as a function of $x$ for fixed $k_n$. 3) Plotting the 3d-plot $(x,k_n) \mapsto \widehat{\gamma}_{k_n}(x)$.



## How to run the program:

### Inputs:
The program takes *either* a time series csv file, *or* specified stock tickers:

-s --stocks: string of two stock tickers, e.g. "AAPL,MSFT"

-fd --from_date: date from which to download stock price history, e.g. "2000-01-01"

-td --to_date: date to which to download stock price history, e.g. "2024-11-01"

-t --transform_type: applies specified transformation to timeseries. Current options: log_diff

-f --file_path: location of time series csv file with 3 columns: time, covariate, rv.

### Outputs:
Output.html file

### Examples:
python3 conditional_hill_estimator.py -s "AAPL,MSFT" -fd "2000-01-01" -td "2024-11-01" -t log_diff

python3 conditional_hill_estimator.py -s "TSLA,NVDA" -t log_diff

python3 conditional_hill_estimator.py -f "time_series.csv" -t log_diff

python3 conditional_hill_estimator.py -f "time_series.csv"
