# Holdings input and concentration checks

The adapter is `src/wharton_engine/holdings.py`, and the gate rule kind is `concentration`. It implements MODEL_READINESS_PLAN.md section 3.

## Files

`holdings.csv`: one row per instrument in each broad sleeve. Weights within a sleeve must sum to 1.

| column | meaning |
|---|---|
| `sleeve` | `equity`, `treasury` or `cash` (the simulated sleeves) |
| `security_id`, `name` | identifier (ticker/CUSIP/ISIN) and name |
| `weight_in_sleeve` | decimal weight of this instrument inside its sleeve |
| `asset_class` | provider asset class |
| `instrument_type` | `stock`, `fund`, `bond`, `bill` or `cash` |
| `issuer`, `sector` | blank or `unknown` means unknown (never treated as zero exposure) |
| `effective_date`, `source` | classification date and provider |

`lookthrough.csv`: constituents of each `fund` row: `fund_id, constituent_id, constituent_name, instrument_type, issuer, sector, weight_in_fund, as_of, source`. Constituent weights that sum to less than 1 leave an unknown residual.

Point `data.holdings_file` and `data.lookthrough_file` at these files. `data/demo/` contains SYNTHETIC examples with made-up identifiers.

## Calculation

Exposure to a constituent as a share of the total portfolio is `sleeve weight × weight_in_sleeve × weight_in_fund`. These states are checked:

- every planned annual target allocation, 2027–2032
- the derisk locked allocation, if the candidate can trigger it
- the post-2033 retained allocation, assessed as if it were the whole portfolio (a conservative upper bound, since the matched reserve holds only Treasuries)

| result | condition |
|---|---|
| FAIL | known exposure to one issuer's stock > `stock_limit`, or to one in-scope sector > `sector_limit` |
| NEEDS_MORE_EVIDENCE | no holdings supplied; a fund without (complete) look-through; unknown issuer/sector mass; an unmapped sleeve; or `drift_check: required` (between-rebalance drift is not simulated) |
| PASS | every state is within limits and all in-scope exposure is identified |

The policy rule's `params` hold `stock_limit` (0.05), `sector_limit` (0.20), `sector_scope` (instrument types counted toward sector limits, default `[stock]`), `exemptions` (empty; add one only with documented provenance) and `interpretation_status`.

## Not yet covered

- The instrument mapping does not change the simulated return proxies or costs. The report says this explicitly.
- Prices, lot sizes, eligibility and competition-account trading mechanics are not implemented (plan sections 3–4).
- Drift between annual rebalances would need a monthly or daily portfolio simulator (plan section 5).
