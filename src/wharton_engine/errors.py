"""Explicit exception types. A numerical bug must never become a market-loss path."""


class EngineError(Exception):
    """Base class for all engine errors."""


class ConfigError(EngineError):
    """Invalid or inconsistent configuration."""


class MissingDataError(EngineError):
    """A required data file or field is absent. Never silently replaced by demo data."""


class DataValidationError(EngineError):
    """Input data fails a validation rule (duplicates, units, nonfinite values...)."""


class CurveError(EngineError):
    """Invalid yield curve or discount factors."""


class NumericalModelError(EngineError):
    """A model produced nonfinite or numerically invalid output (not a market outcome)."""


class CashFlowError(EngineError):
    """Accounting invariants violated (negative balances, double counting, ...)."""


class BudgetExceededError(EngineError):
    """Runtime or path budget exhausted."""


class VisibilityError(EngineError):
    """A policy or projection attempted to access hidden future information."""
