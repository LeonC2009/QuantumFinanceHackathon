"""
Dummy trade-execution layer for the antipodal hedge pipeline.

Design note: real energy-futures trading (ICE, CME) requires a regulated
FCM brokerage account and KYC -- not feasible inside a hackathon, and not
what judges expect to see. This module implements a paper-trading loop
instead: a synthetic price feed plus an in-memory broker that fills orders
instantly at the simulated price. rebalance_to_target() is deliberately
solver-agnostic -- it just needs a {'long': [...], 'short': [...]} dict, so
it doesn't matter whether that came from brute_force_best() or a decoded
QAOA/annealer result. Swapping in a real paper-trading sandbox (e.g.
Alpaca's) later is a drop-in replacement of DummyBroker, not a rewrite.

Pipeline this module closes the loop on:
    market/ESG data -> QUBO -> (quantum or classical) solver -> decision
    -> [THIS MODULE] -> executed trades -> portfolio + carbon tracking
"""

import numpy as np
from cardinality_qubo import names, returns, vols, carbon, brute_force_best

carbon_by_name = dict(zip(names, carbon))


class SyntheticPriceFeed:
    """Generates a daily synthetic price path per asset via geometric
    Brownian motion, seeded from the return/vol assumptions already
    defined in cardinality_qubo.py -- so the "market" this trades against
    is consistent with the numbers the solver optimized for."""

    def __init__(self, start_price=100.0, seed=42):
        self.prices = {n: start_price for n in names}
        self.rng = np.random.default_rng(seed)

    def step(self, dt=1 / 252):
        for i, name in enumerate(names):
            mu, sigma = returns[i], vols[i]
            shock = self.rng.normal(0, 1)
            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * np.sqrt(dt) * shock
            self.prices[name] *= np.exp(drift + diffusion)
        return dict(self.prices)


class DummyBroker:
    """In-memory paper-trading broker. Fills every order instantly at the
    quoted price -- no slippage/latency modeling, intentionally, since the
    point of the demo is the decision logic, not execution microstructure."""

    def __init__(self, starting_cash=1_000_000.0):
        self.cash = starting_cash
        self.positions = {n: 0.0 for n in names}  # units held; negative = short
        self.trade_log = []

    def submit_order(self, asset, quantity, price):
        """quantity > 0: buy/go long that many units. quantity < 0: sell/short."""
        self.cash -= quantity * price
        self.positions[asset] += quantity
        self.trade_log.append({"asset": asset, "quantity": quantity, "price": price})

    def portfolio_value(self, prices):
        return self.cash + sum(self.positions[n] * prices[n] for n in names)

    def carbon_exposure(self, current_carbon=None):
        """current_carbon: optional {name: carbon_intensity} dict reflecting
        today's (possibly drifted) grid-mix data. Falls back to the static
        baseline for standalone use (e.g. this module's own demo below),
        but the simulation should always pass the live vector -- otherwise
        this silently reports exposure against stale carbon numbers even
        as the held positions' actual carbon intensity has moved."""
        source = current_carbon if current_carbon is not None else carbon_by_name
        return sum(self.positions[n] * source[n] for n in names)


def rebalance_to_target(broker, decision, prices, notional_per_leg=100_000.0):
    """decision: {'long': [names...], 'short': [names...]}, as returned by
    brute_force_best() or decoded from a QUBO solution -- this function
    doesn't care which solver produced it. Builds an equal-notional target
    book and trades only the delta from current positions, so calling this
    repeatedly as the solver's answer changes over time is a real rebalance,
    not a full liquidate-and-rebuy each time."""
    target = {n: 0.0 for n in names}
    for n in decision["long"]:
        target[n] = notional_per_leg / prices[n]
    for n in decision["short"]:
        target[n] = -notional_per_leg / prices[n]

    for n in names:
        delta = target[n] - broker.positions[n]
        if abs(delta) > 1e-6:
            broker.submit_order(n, delta, prices[n])
    return target


if __name__ == "__main__":
    feed = SyntheticPriceFeed()
    broker = DummyBroker()

    decision, _ = brute_force_best(k_long=2, k_short=2)
    print(f"Decision -> long {decision['long']}, short {decision['short']}\n")

    print(f"{'Day':>4} {'Portfolio Value':>18} {'Carbon Exposure':>18}")
    for day in range(1, 11):
        prices = feed.step()
        if day == 1:
            rebalance_to_target(broker, decision, prices)
        val = broker.portfolio_value(prices)
        carb = broker.carbon_exposure()
        print(f"{day:>4} {val:>18,.2f} {carb:>18,.1f}")

    print(f"\nTotal trades executed: {len(broker.trade_log)}")
