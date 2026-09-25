"""
Full antipodal-hedge simulation: drifting grid-mix (carbon) data, a periodic
re-solve of the QUBO/Borsuk-Ulam decision, rebalancing trades, and a JSON log
of everything -- portfolio value, carbon exposure, current hedge legs, and
trades -- for the visualization dashboard to consume.

This is the piece that makes the theorem's role visible over time: as the
carbon vector drifts, the antipodal pair it maps to can change, and you see
the hedge legs rotate in response instead of staying fixed.
"""

import json
import numpy as np
from cardinality_qubo import names, returns, vols, carbon, brute_force_best
from trade_executor import SyntheticPriceFeed, DummyBroker, rebalance_to_target


class CarbonDrift:
    """Random-walks each asset's carbon intensity a little each day, standing
    in for real grid-mix shifts (more/less wind or coal on a given day).
    Floored at 1.0 so it can't go negative or degenerate."""

    def __init__(self, base_carbon, seed=7, daily_vol=0.03):
        self.carbon = {n: float(c) for n, c in zip(names, base_carbon)}
        self.rng = np.random.default_rng(seed)
        self.daily_vol = daily_vol

    def step(self):
        for n in names:
            shock = self.rng.normal(0, self.daily_vol)
            self.carbon[n] = max(1.0, self.carbon[n] * (1 + shock))
        return dict(self.carbon)

    def as_vector(self):
        return np.array([self.carbon[n] for n in names])


def run_simulation(n_days=60, resolve_every=5, k_long=2, k_short=2,
                    notional_per_leg=100_000.0, out_path="simulation_log.json"):
    feed = SyntheticPriceFeed()
    broker = DummyBroker()
    drift = CarbonDrift(carbon)

    log = []
    decision = None

    for day in range(1, n_days + 1):
        prices = feed.step()
        current_carbon = drift.step()

        # Re-solve on the schedule, using the live (drifted) carbon vector
        trades_before = len(broker.trade_log)
        if decision is None or day % resolve_every == 0:
            decision, _ = brute_force_best(
                k_long=k_long, k_short=k_short,
                carbon_vec=drift.as_vector(),
            )
            rebalance_to_target(broker, decision, prices, notional_per_leg)
            rebalanced = True
        else:
            rebalanced = False
        n_trades_today = len(broker.trade_log) - trades_before

        log.append({
            "day": day,
            "portfolio_value": broker.portfolio_value(prices),
            "carbon_exposure": broker.carbon_exposure(current_carbon),
            "long": decision["long"],
            "short": decision["short"],
            "rebalanced": rebalanced,
            "n_trades_today": n_trades_today,
        })

    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"Simulated {n_days} days, re-solving every {resolve_every} days.")
    print(f"Total rebalances: {sum(1 for r in log if r['rebalanced'])}")
    print(f"Final portfolio value: {log[-1]['portfolio_value']:,.2f}")
    print(f"Final carbon exposure: {log[-1]['carbon_exposure']:,.1f}")
    print(f"Log written to {out_path}")
    return log


if __name__ == "__main__":
    run_simulation()
