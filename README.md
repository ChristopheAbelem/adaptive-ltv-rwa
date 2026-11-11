# Adaptive LTV for tokenized real estate

A fuzzy closed-loop controller that sets the maximum loan-to-value (LTV) of a lending market backed by tokenized
real estate, tested against the fixed 60% LTV of RealT's RMM in a market shock and in an operational collapse
(rent and pool liquidity failing while prices stay high, as with RealT in 2025-2026).

> Status: in progress. The one-sentence answer and the limitations will open this README once the simulations run.

## Structure

| Path | Content |
| --- | --- |
| `adaptive_ltv/` | Package: input signals, fuzzy variables, rules, market environment, closed-loop simulation |
| `tests/` | `pytest` checks (de-smoothing, membership coverage, rule base, closed forms) |
| `00_data_test.ipynb` | Downloads and freezes every input; pass/fail per source |
| `01_adaptive_ltv.ipynb` | Report: modelling, rules, tests, surfaces, simulations, conclusions |

## Data

| Input | Source | Status |
| --- | --- | --- |
| Detroit house prices | FRED `DEXRNSA` (S&P Cotality Case-Shiller), downloaded at runtime, not redistributed | real |
| Pool utilization | DefiLlama, RealT RMM | real |
| Risk parameters, liquidations | RMM v3 contracts on Gnosis Chain (public RPC) | real |
| Rent coverage, appraisal steps | Stylized from documented dates (no public time series) | stylized |

## Run

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

Then run `00_data_test.ipynb` once and `01_adaptive_ltv.ipynb` top to bottom.
