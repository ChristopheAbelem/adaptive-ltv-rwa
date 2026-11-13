"""Market environment for the closed loop: next_state = update_market(state, max_ltv).

One step = one month. The pool lends stablecoins to borrowers who post tokenized property as collateral.
The controller's action (max LTV) changes how much borrowers may owe. Two of its inputs depend on that: the pool's
utilization (borrowed / supplied) and the volatility of the collateral's market price, which falls when liquidations
force sales. Both are read again next month: that is the feedback loop.

All behavioural parameters are stated assumptions (see ASSUMPTIONS); the price, volatility and rent paths come from
the data (Detroit index, stylized rent).
"""

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

LIQ_THRESHOLD = 0.70          # RealT RMM liquidation threshold (on-chain, data/raw/rmm_params_latest.csv)
LIQ_BONUS = 0.10              # RealT RMM liquidation bonus (on-chain)
START_LTV = 0.60              # RealT RMM max LTV (on-chain): the loan book both policies start from

ASSUMPTIONS = {
    "n_borrowers": (200, "number of borrower positions, each with one unit of collateral"),
    "target_share": ((0.5, 1.0), "each borrower borrows a fixed share of the allowed amount, drawn uniformly"),
    "deleverage_rate": (0.10, "share of the excess over the new limit repaid each month when the limit falls (sensitivity: 0 to 20%)"),
    "initial_utilization": (0.85, "pool utilization at the start (2024-2025 level, DefiLlama)"),
    "rate_kink": ((0.90, 0.05, 0.60), "interest model (Aave-style stablecoin curve): kink at 90% utilization, 5% a year at the kink, 65% at 100%"),
    "lender_flows": ((0.01, 0.10, 0.17), "calibrated on DefiLlama 2024-2026: deposits and withdrawals balance (1% a month each) "
                     "while rent is paid, so utilization stays near 85%; once rent coverage falls to 10% or less, lenders run: "
                     "they try to withdraw 17% of the supply a month (the largest observed fall, Feb 2026), limited by idle cash"),
    "token_discount": (0.60, "with zero rent, a token sells 60% below the house value it represents"),
    "price_impact": (0.50, "liquidation sales lower the sale price by 0.5 x (collateral sold that month / all collateral at the start)"),
    "appraisal_months": (12, "the pool's price (oracle) is the appraisal, updated every 12 months"),
}


@dataclass
class State:
    month: int
    house_index: float            # Detroit house value, normalized (data path)
    oracle_price: float           # appraisal the pool uses to check loans
    token_value: float            # what a token actually sells for (house value x rent discount)
    volatility: float             # controller input 1 (endogenous): volatility of the collateral's market price
    rent_coverage: float          # controller input 3 (exogenous: scenario path)
    supply: float                 # stablecoins deposited by lenders
    debt: np.ndarray              # debt per position
    collateral: np.ndarray        # collateral units per position (0 after liquidation)
    target_share: np.ndarray      # behaviour per position
    liquidations: int = 0         # liquidations this month
    bad_debt: float = 0.0         # bad debt realized this month
    market_price: float = 1.0     # price a token actually sells for this month (token value minus selling pressure)
    price_returns: list = field(default_factory=list)   # monthly log returns of the market price (last 12 used)

    @property
    def total_debt(self) -> float:
        return float(self.debt.sum())

    @property
    def utilization(self) -> float:
        """Controller input 2 (endogenous): borrowed / supplied."""
        return float(np.clip(self.total_debt / self.supply, 0, 1)) if self.supply > 0 else 1.0

    @property
    def idle_cash(self) -> float:
        return max(self.supply - self.total_debt, 0.0)

    def latent_bad_debt(self) -> float:
        """Debt that the collateral could not cover if every open loan were closed today."""
        return float(np.maximum(self.debt - self.collateral * self.token_value, 0).sum())


def borrow_rate(utilization: float, kink=0.90, base=0.05, top=0.60) -> float:
    """Annual borrow rate, Aave-style: gentle slope up to the kink, steep above it."""
    if utilization <= kink:
        return base * utilization / kink
    return base + top * (utilization - kink) / (1 - kink)


@dataclass
class Scenario:
    """Exogenous paths, one value per month."""

    name: str
    house_index: pd.Series        # normalized to 1.0 at the start
    volatility: pd.Series         # data volatility (reference only: the controller reads the simulated one)
    rent_coverage: pd.Series
    pre_returns: pd.Series        # the 12 monthly house returns before the start (to start the rolling volatility)

    def __len__(self):
        return len(self.house_index)


def initial_state(scenario: Scenario, max_ltv: float, seed: int = 0) -> State:
    rng = np.random.default_rng(seed)
    n = ASSUMPTIONS["n_borrowers"][0]
    lo, hi = ASSUMPTIONS["target_share"][0]
    share = rng.uniform(lo, hi, n)
    p0 = float(scenario.house_index.iloc[0])
    debt = share * max_ltv * p0
    supply = debt.sum() / ASSUMPTIONS["initial_utilization"][0]
    r0 = float(scenario.rent_coverage.iloc[0])
    token = p0 * (1 - ASSUMPTIONS["token_discount"][0] * (1 - r0))
    pre = list(scenario.pre_returns.iloc[-12:].astype(float))
    return State(0, p0, p0, token, rolling_vol(pre), r0, supply, debt, np.ones(n), share,
                 market_price=token, price_returns=pre)


def rolling_vol(returns: list, window: int = 12) -> float:
    """Annualized standard deviation of the last `window` monthly log returns (same as section 2.1)."""
    r = np.asarray(returns[-window:], dtype=float)
    return float(r.std(ddof=1) * np.sqrt(12))


def update_market(state: State, max_ltv: float, scenario: Scenario) -> State:
    """One month: borrowers react to the limit, interest accrues, lenders move, prices move, liquidations."""
    s = replace(state, debt=state.debt.copy(), collateral=state.collateral.copy())
    t = s.month + 1
    open_ = s.collateral > 0

    # 1. borrowers react to the controller's limit (checked against the pool's price, the appraisal)
    allowed = max_ltv * s.oracle_price * s.collateral
    target = s.target_share * allowed
    excess = np.where(open_, np.maximum(s.debt - allowed, 0), 0)
    s.debt -= ASSUMPTIONS["deleverage_rate"][0] * excess                      # repay part of the excess
    want = np.where(open_, np.maximum(target - s.debt, 0), 0)                  # new borrowing wanted
    if want.sum() > 0:
        s.debt += want * min(1.0, s.idle_cash / want.sum())                    # limited by idle cash

    # 2. interest: borrowers pay it out of the rent their tokens earn; with rent coverage R, the share 1 - R that
    #    rent cannot pay is added to the debt (a modelling rule, not a tunable assumption)
    rate = borrow_rate(s.utilization, *ASSUMPTIONS["rate_kink"][0]) / 12
    s.debt *= 1 + rate * (1 - s.rent_coverage)

    # 3. lenders deposit and withdraw (withdrawals limited by idle cash)
    dep, run_trigger, run_rate = ASSUMPTIONS["lender_flows"][0]
    s.rent_coverage = float(scenario.rent_coverage.iloc[t])
    wanted = dep + (run_rate if s.rent_coverage <= run_trigger else 0.0)
    withdraw = min(s.supply * wanted, s.idle_cash)
    s.supply += s.supply * dep - withdraw

    # 4. prices move (data paths); the appraisal is refreshed every N months
    s.house_index = float(scenario.house_index.iloc[t])
    if t % ASSUMPTIONS["appraisal_months"][0] == 0:
        s.oracle_price = s.house_index
    s.token_value = s.house_index * (1 - ASSUMPTIONS["token_discount"][0] * (1 - s.rent_coverage))

    # 5. liquidations: loans above the threshold (on the appraisal) are closed by selling the collateral
    breach = (s.collateral > 0) & (s.debt > LIQ_THRESHOLD * s.oracle_price * s.collateral)
    sold = s.collateral[breach].sum()
    impact = ASSUMPTIONS["price_impact"][0] * sold / len(s.collateral)
    sale_price = s.token_value * (1 - min(impact, 0.9))
    proceeds = s.collateral[breach] * sale_price / (1 + LIQ_BONUS)            # the liquidator keeps the bonus
    repaid = np.minimum(s.debt[breach], proceeds)
    bad = float((s.debt[breach] - repaid).sum())
    s.debt[breach] = 0.0
    s.collateral[breach] = 0.0
    s.supply -= bad                                                            # lenders absorb the loss

    # 6. the market price (token value minus selling pressure) feeds the volatility the controller reads next month
    new_price = s.token_value * (1 - min(impact, 0.9))
    s.price_returns = s.price_returns + [float(np.log(new_price / s.market_price))]
    s.market_price = new_price
    s.volatility = rolling_vol(s.price_returns)
    s.liquidations, s.bad_debt, s.month = int(breach.sum()), bad, t
    return s


def run(scenario: Scenario, policy, seed: int = 0, on_step=None) -> pd.DataFrame:
    """Closed loop: policy(volatility, utilization, rent_coverage) -> max LTV, every month.

    on_step(row) is called after each month (used for the real-time display)."""
    s = initial_state(scenario, START_LTV, seed)     # both policies inherit the same book, built under RealT's 60%
    s0_supply = s.supply
    rows = []
    for _ in range(len(scenario) - 1):
        ltv = policy(s.volatility, s.utilization, s.rent_coverage)
        s = update_market(s, ltv, scenario)
        row = {"date": scenario.house_index.index[s.month], "max_ltv": ltv, "volatility": s.volatility,
               "market_price": s.market_price,
               "utilization": s.utilization, "rent_coverage": s.rent_coverage, "house_index": s.house_index,
               "oracle_price": s.oracle_price, "token_value": s.token_value, "total_debt": s.total_debt,
               "supply": s.supply, "liquidations": s.liquidations, "bad_debt": s.bad_debt,
               "latent_bad_debt": s.latent_bad_debt()}
        rows.append(row)
        if on_step:
            on_step(row)
    out = pd.DataFrame(rows).set_index("date")
    out["cum_bad_debt"] = out["bad_debt"].cumsum()
    out["loss_to_date"] = out["cum_bad_debt"] + out["latent_bad_debt"]
    out.attrs["initial_supply"] = s0_supply
    return out


def fixed_policy(ltv: float = 0.60):
    """RealT's rule: the same limit whatever happens."""
    return lambda volatility, utilization, rent_coverage: ltv


def loss_summary(result: pd.DataFrame, initial_supply: float | None = None) -> dict:
    """Realized bad debt + debt the collateral cannot cover at the end, as % of the initial lender supply."""
    initial_supply = initial_supply or result.attrs["initial_supply"]
    realized = result["bad_debt"].sum()
    latent = result["latent_bad_debt"].iloc[-1]
    return {"liquidations": int(result["liquidations"].sum()),
            "realized_bad_debt_%": 100 * realized / initial_supply,
            "uncovered_debt_at_end_%": 100 * latent / initial_supply,
            "total_loss_%": 100 * (realized + latent) / initial_supply,
            "lowest_max_ltv_%": 100 * result["max_ltv"].min()}


class assumption:
    """Temporarily change one assumption (sensitivity analysis): `with assumption("deleverage_rate", 0.0): ...`.

    ASSUMPTIONS is module-level state: runs inside the block use the new value, and it is restored on exit."""

    def __init__(self, key: str, value):
        if key not in ASSUMPTIONS:
            raise KeyError(f"unknown assumption {key!r}; known: {sorted(ASSUMPTIONS)}")
        self.key, self.value = key, value

    def __enter__(self):
        self.old = ASSUMPTIONS[self.key]
        ASSUMPTIONS[self.key] = (self.value, self.old[1])

    def __exit__(self, *exc):
        ASSUMPTIONS[self.key] = self.old
