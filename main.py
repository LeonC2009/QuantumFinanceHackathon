import numpy as np

assets = {
    "Solar_US_East":   [0.08, 15],
    "Wind_EU_North":   [0.07, 11],
    "Gas_Futures_US":  [0.08, 450],
    "Coal_Power_Asia": [0.12, 820],
}
returns = np.array([a[0] for a in assets.values()])
carbon  = np.array([a[1] for a in assets.values()])

# Mock covariance matrix (clean assets slightly correlated with each other,
# dirty assets correlated with each other, clean/dirty weakly correlated)
cov = np.array([
    [0.014, 0.010, 0.002, 0.001],
    [0.010, 0.011, 0.001, 0.002],
    [0.002, 0.001, 0.019, 0.013],
    [0.001, 0.002, 0.013, 0.045],
])

def portfolio_return(w):   return w @ returns
def portfolio_variance(w): return w @ cov @ w      # quadratic -> even automatically
def portfolio_carbon(w):   return w @ carbon
