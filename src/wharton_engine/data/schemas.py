"""Canonical input schemas (documented in docs/financial_conventions.md)."""

MONTHLY_COLUMNS = {
    "date": "month-end observation date, YYYY-MM-DD",
    "available_date": "date the observation became publicly available (blank -> release-lag rule)",
    "equity_tr": "broad-equity simple TOTAL return for the month, decimal (0.01 = 1%)",
    "treasury_tr": "rolling intermediate-Treasury simple TOTAL return for the month, decimal",
    "cash_tr": "cash / short-bill simple total return for the month, decimal",
    "cpi_index": "CPI index level at month end (positive)",
}

ANNUAL_COLUMNS = {
    "year": "calendar year of the return interval",
    "available_date": "date the full-year observation became available",
    "equity_tr": "calendar-year broad-equity simple total return, decimal",
    "treasury_tr": "calendar-year rolling-Treasury simple total return, decimal",
    "cash_tr": "calendar-year cash / bill simple total return, decimal",
    "inflation": "calendar-year CPI growth (Dec/Dec), decimal",
}

RETURN_TYPES_ALLOWED = {"total_return"}
DATA_KINDS = {"market_history", "synthetic", "test_fixture"}

MANIFEST_REQUIRED = ["data_kind", "description", "series", "limitations"]
SERIES_REQUIRED = ["name", "return_type", "source"]
