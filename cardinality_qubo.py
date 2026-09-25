"""
Cardinality-constrained antipodal hedge search, formulated as a QUBO.

Extends the earlier evaluate_portfolio() idea: instead of searching continuous
weights on the unit sphere, we restrict to a fixed number of long and short
positions with equal magnitude. This turns the antipodal search into a
combinatorial selection problem (which assets go long / short), which is
where the exponential blowup -- and the actual case for quantum -- comes from.

Weight model:
    w_i = mag * (x_i - y_i),   mag = 1 / sqrt(k_long + k_short)
    x_i, y_i in {0, 1}         (x_i=1: asset i is long, y_i=1: asset i is short)

Constraints (enforced as penalty terms so this stays a QUBO):
    x_i + y_i <= 1              for every i   (can't be long AND short)
    sum(x) == k_long
    sum(y) == k_short

Objective (minimize):
    variance(w)                                    -- risk
    + lambda_ret  * return(w)^2                     -- push toward market-neutral
    + lambda_esg  * carbon_separation(w)            -- push carbon as negative
                                                          as possible (clean long,
                                                          dirty short)
"""

import itertools
import numpy as np

# ---------------------------------------------------------------------------
# 1. Asset universe: [expected return, volatility, carbon intensity gCO2/kWh]
# ---------------------------------------------------------------------------
assets = {
    "Solar_US_East":        [0.08, 0.11,  15],
    "Wind_EU_North":        [0.07, 0.10,  11],
    "Hydro_Nordic":         [0.05, 0.07,   4],
    "Geothermal_Iceland":   [0.06, 0.09,   8],
    "Gas_Futures_US":       [0.08, 0.14, 450],
    "Coal_Power_Asia":      [0.12, 0.22, 820],
    "Oil_Futures_Brent":    [0.10, 0.18, 650],
    "LNG_Shipping_Futures": [0.09, 0.16, 500],
}
names   = list(assets.keys())
n       = len(names)
returns = np.array([v[0] for v in assets.values()])
vols    = np.array([v[1] for v in assets.values()])
carbon  = np.array([v[2] for v in assets.values()])

# Mock covariance: clean assets (idx 0-3) correlated with each other,
# dirty assets (idx 4-7) correlated with each other, weak cross-correlation.
corr = np.full((n, n), 0.10)
corr[:4, :4] = 0.60
corr[4:, 4:] = 0.60
np.fill_diagonal(corr, 1.0)
cov = np.outer(vols, vols) * corr

# ---------------------------------------------------------------------------
# 2. Classical brute-force baseline (only tractable because n=8 is tiny)
# ---------------------------------------------------------------------------
def brute_force_best(k_long=2, k_short=2, lambda_ret=50.0, lambda_esg=1.0,
                       carbon_vec=None):
    """Enumerate every valid long/short split and return the best one.
    Number of candidates = C(n, k_long+k_short) * C(k_long+k_short, k_long).
    This is the number that explodes as the asset universe grows -- and is
    exactly the search space a QAOA/annealing solver would explore instead.

    carbon_vec: optional override for the carbon-intensity vector (used by
    the simulation to re-solve against live-updating grid-mix data instead
    of the static baseline).
    """
    if carbon_vec is None:
        carbon_vec = carbon
    k = k_long + k_short
    mag = 1.0 / np.sqrt(k)
    best = None
    n_candidates = 0

    for combo in itertools.combinations(range(n), k):
        for long_idx in itertools.combinations(combo, k_long):
            n_candidates += 1
            short_idx = [i for i in combo if i not in long_idx]
            w = np.zeros(n)
            w[list(long_idx)] = mag
            w[short_idx] = -mag

            ret  = w @ returns
            var  = w @ cov @ w
            carb = w @ carbon_vec
            score = var + lambda_ret * ret**2 + lambda_esg * carb

            if best is None or score < best["score"]:
                best = {
                    "score": score, "weights": w.copy(),
                    "long": [names[i] for i in long_idx],
                    "short": [names[i] for i in short_idx],
                    "return": ret, "variance": var, "carbon": carb,
                }
    return best, n_candidates


# ---------------------------------------------------------------------------
# 3. QUBO builder -- the quantum-ready version of the same problem
# ---------------------------------------------------------------------------
def build_qubo(k_long=2, k_short=2, lambda_ret=50.0, lambda_esg=1.0,
                penalty=2000.0):
    """Build Q such that x^T Q x (x = concatenated [x_0..x_{n-1}, y_0..y_{n-1}])
    equals the objective above plus constraint penalties. Returned as a dict
    {(i, j): coeff} -- the standard format expected by dimod/D-Wave/QAOA
    QUBO-to-Ising converters (Qiskit's QuadraticProgram.from_qubo can also
    ingest this directly).
    """
    k = k_long + k_short
    mag = 1.0 / np.sqrt(k)
    N = 2 * n  # x_0..x_{n-1}, then y_0..y_{n-1}
    Q = {}

    def add(i, j, val):
        key = (min(i, j), max(i, j))
        Q[key] = Q.get(key, 0.0) + val

    # --- variance term: mag^2 * (x-y)^T cov (x-y) ---
    for i in range(n):
        for j in range(n):
            c = (mag ** 2) * cov[i, j]
            add(i, j, c)              # x_i x_j
            add(n + i, n + j, c)      # y_i y_j
            add(i, n + j, -2 * c)     # x_i y_j cross term

    # --- return^2 term: lambda_ret * (mag * sum r_i(x_i - y_i))^2 ---
    for i in range(n):
        for j in range(n):
            c = lambda_ret * (mag ** 2) * returns[i] * returns[j]
            add(i, j, c)
            add(n + i, n + j, c)
            add(i, n + j, -2 * c)

    # --- carbon term (linear -> diagonal of Q): lambda_esg * mag * c_i * (x_i - y_i) ---
    for i in range(n):
        add(i, i, lambda_esg * mag * carbon[i])
        add(n + i, n + i, -lambda_esg * mag * carbon[i])

    # --- penalty: (x_i + y_i - <=1) handled as x_i*y_i penalty (can't both be 1) ---
    for i in range(n):
        add(i, n + i, penalty)

    # --- penalty: (sum(x) - k_long)^2 ---
    for i in range(n):
        add(i, i, penalty * (1 - 2 * k_long))
        for j in range(n):
            if i != j:
                add(i, j, penalty)

    # --- penalty: (sum(y) - k_short)^2 ---
    for i in range(n):
        add(n + i, n + i, penalty * (1 - 2 * k_short))
        for j in range(n):
            if i != j:
                add(n + i, n + j, penalty)

    return Q, N


if __name__ == "__main__":
    best, n_candidates = brute_force_best(k_long=2, k_short=2)
    print(f"Search space size: {n_candidates} candidates (n={n} assets)")
    print(f"Best long:  {best['long']}")
    print(f"Best short: {best['short']}")
    print(f"Return:   {best['return']:.4f}")
    print(f"Variance: {best['variance']:.4f}")
    print(f"Carbon exposure: {best['carbon']:.1f} gCO2/kWh (more negative = better)")

    Q, N = build_qubo(k_long=2, k_short=2)
    print(f"\nQUBO built: {N} binary variables, {len(Q)} nonzero terms")
    print("This Q dict is ready to hand to a QAOA solver (via Qiskit's "
          "QuadraticProgram.from_qubo) or a quantum annealer (dimod/D-Wave).")
