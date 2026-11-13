# Adaptive LTV for tokenized real estate

A fuzzy closed-loop controller that sets the maximum loan-to-value (LTV) of a lending market backed by tokenized
real estate, tested against the fixed 60% LTV of RealT's RMM in a market shock (Detroit 2006-2012) and in an
operational collapse (RealT 2024-2026: rent and pool liquidity fail while house prices stay high).

**Result.** In the operational collapse the controller cuts lenders' losses from about 46% to about 6% of deposits,
provided borrowers repay part of their loans when the limit falls; with no repayment it changes nothing. In the
market shock losses are small under both rules and the comparison is not robust to the assumptions.

> Status: course deliverable in progress (real-time app and conclusions pending).

## Structure

| Path | Content |
| --- | --- |
| `adaptive_ltv/inputs.py` | Input signals: de-smoothed Detroit volatility, pool utilization, stylized rent coverage |
| `adaptive_ltv/membership.py` | Fuzzy variables and membership functions |
| `adaptive_ltv/controller.py` | 27-rule Mamdani controller (severity scoring), exposes every inference step |
| `adaptive_ltv/analysis.py` | Static test cases and control surfaces |
| `adaptive_ltv/environment.py` | Monthly lending-pool environment, `update_market(state, max_ltv)`, closed-loop `run` |
| `adaptive_ltv/scenarios.py` | Market shock, operational collapse, Merton random futures |
| `adaptive_ltv/merton.py` | Merton jump-diffusion: calibration, simulation, closed forms |
| `adaptive_ltv/plots.py` | Chart helpers |
| `tests/` | `pytest` suite (de-smoothing, coverage, rules, environment accounting, Merton closed forms) |
| `00_data_test.ipynb` | Downloads and freezes every input; pass/fail per source |
| `01_adaptive_ltv.ipynb` | Report: modelling, rules, surfaces, environment, closed-loop results, sensitivity |

## Data

| Input | Source | Status |
| --- | --- | --- |
| Detroit house prices | FRED `DEXRNSA` (S&P Cotality Case-Shiller), downloaded at runtime, not redistributed | real |
| Pool utilization, lender flows | DefiLlama, RealT RMM | real |
| Risk parameters, liquidations | RMM v3 contracts on Gnosis Chain (public RPC) | real |
| Rent coverage, appraisal steps | Stylized from documented dates (no public time series) | stylized |

## Run

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

Then run `00_data_test.ipynb` once and `01_adaptive_ltv.ipynb` top to bottom.
