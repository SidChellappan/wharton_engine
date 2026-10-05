# Real-data input folder

Place documented market history here for a decision-support run. Nothing in this folder is shipped
with the engine. The run stops with `MissingDataError` if a file is missing; it never falls back to
the synthetic demo fixture.

Required files (schemas in `docs/financial_conventions.md`):

| file | content |
|---|---|
| `monthly_returns.csv` | `date, available_date, equity_tr, treasury_tr, cash_tr, cpi_index` (decimal simple TOTAL returns; month-end dates) |
| `curve_snapshot.csv` | `date, available_date, tenor_years, quote_type (zero/par), rate, rate_units (decimal/percent), compounding (annual), source` |
| `SOURCE_MANIFEST.yaml` | `data_kind: market_history`, a description, `series` (equity/treasury/cash/cpi with `name`, `return_type: total_return`, `source`), `limitations`, licensing notes |

`tests/fixtures/real_schema/` contains a small fixture in exactly this schema (labelled `test_fixture`).
Copy its layout, replace the numbers with your sourced data, and run:

    wharton-engine run --config configs/decision_template.yaml
