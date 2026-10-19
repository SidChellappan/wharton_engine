"""Retrospective monthly-origin diagnostics, deliberately separate from selection.

Forecasts use strictly available joint monthly observations. Targets use complete
asset-return months and directly observed CPI endpoints. Adjacent targets overlap;
resampling intervals are sensitivity diagnostics, never a new PASS mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import DataValidationError, MissingDataError, NumericalModelError
from . import bayesian as B
from .bootstrap import valid_block_starts, year_segments
from .validation import crps_samples

RETURNS = ["equity_tr", "treasury_tr", "cash_tr"]
SERIES = list(B.SERIES)


@dataclass(frozen=True)
class MonthlyModel:
    name: str
    kind: str
    training_months: int
    block_months: int = 0

    def __post_init__(self):
        if self.kind not in {"niw", "fixed", "bootstrap", "dynamic", "yield_var", "yield_trend", "yield_ensemble"} or self.training_months < 24:
            raise ValueError("invalid monthly model or training minimum")
        if self.kind == "bootstrap" and not 1 <= self.block_months <= self.training_months:
            raise ValueError("bootstrap needs a positive block no longer than its training minimum")
        if self.kind != "bootstrap" and self.block_months != 0:
            raise ValueError("only bootstrap models use block_months")


def prepare_monthly(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"date", "available_date", "cpi_index", *RETURNS}
    if not required.issubset(frame):
        raise DataValidationError(f"monthly input missing {required - set(frame)}")
    d = frame.copy()
    for col in ("date", "available_date"):
        d[col] = pd.to_datetime(d[col], errors="raise")
        if d[col].isna().any():
            raise DataValidationError(f"missing {col}")
    if d.date.duplicated().any() or d.date.dt.to_period("M").duplicated().any():
        raise DataValidationError("duplicate monthly observations")
    if not d.date.dt.is_month_end.all() or (d.available_date < d.date).any():
        raise DataValidationError("dates must be month ends with no premature availability")
    for col in RETURNS:
        d[col] = pd.to_numeric(d[col], errors="raise")
        if not np.isfinite(d[col]).all() or (d[col] <= -1).any():
            raise DataValidationError(f"invalid {col}; missing returns are not zero")
    d["cpi_index"] = pd.to_numeric(d.cpi_index, errors="raise")
    if ((d.cpi_index <= 0) | np.isinf(d.cpi_index)).any():
        raise DataValidationError("CPI must be positive when observed")
    d = d.sort_values("date").reset_index(drop=True)
    month = d.date.dt.year * 12 + d.date.dt.month
    d["month_id"] = month
    adjacent = month.diff().eq(1)
    d["inflation"] = (d.cpi_index / d.cpi_index.shift() - 1).where(adjacent)
    # Both CPI levels must have been released. Current-row availability is not enough.
    d["joint_available_date"] = pd.concat([d.available_date, d.available_date.shift()], axis=1).max(axis=1)
    d[SERIES[:3]] = d[RETURNS]
    d["joint_complete"] = np.isfinite(d[SERIES]).all(axis=1)
    return d, d.loc[~d.joint_complete, ["date", "cpi_index", "inflation"]].copy()


def training_view(d: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    return d.loc[(d.date < origin) & (d.joint_available_date < origin) & d.joint_complete].copy()


def observed_target(d: pd.DataFrame, origin: pd.Timestamp, horizon: int,
                    as_of: pd.Timestamp) -> tuple[np.ndarray | None, str]:
    if horizon < 1 or origin.day != 1:
        raise ValueError("positive horizon and month-start origin required")
    expected = pd.date_range(origin, periods=horizon, freq="ME")
    indexed = d.set_index("date")
    target = indexed.reindex(expected)
    if target[RETURNS].isna().any().any():
        return None, "missing outcome asset-return month"
    if target.available_date.isna().any() or (target.available_date > as_of).any():
        return None, "outcome not available at experiment cutoff"
    previous = origin - pd.offsets.Day(1)
    if previous not in indexed.index:
        return None, "missing beginning CPI endpoint"
    anchor = indexed.loc[previous]
    if (pd.isna(anchor.cpi_index) or pd.isna(target.iloc[-1].cpi_index)
            or anchor.available_date > as_of):
        return None, "missing or unavailable observed CPI endpoint"
    y = np.r_[np.log1p(target[RETURNS].to_numpy(float)).sum(axis=0),
              np.log(target.iloc[-1].cpi_index / anchor.cpi_index)]
    if not np.isfinite(y).all():
        raise DataValidationError("nonfinite observed cumulative log outcome")
    return y, ""


def monthly_prior(spec) -> B.NIWPrior:
    if spec.empirical_bayes:
        raise ValueError("monthly experiment requires an explicitly fixed prior")
    annual = B.prior_from_spec(spec)
    return B.NIWPrior(annual.mu0 / 12, annual.kappa0, annual.nu0, annual.C0 / 12,
                      False, "monthly iid scaling (mean/12, covariance/12); " + annual.provenance)


def predictive_paths(train: pd.DataFrame, model: MonthlyModel, prior: B.NIWPrior,
                     months: int, n: int, rng: np.random.Generator, fitted=None) -> np.ndarray:
    """Joint log-growth paths, including any unobserved release-lag gap."""
    if len(train) < model.training_months:
        raise MissingDataError("not enough complete, available training months")
    x = np.log1p(train[SERIES].to_numpy(float))
    if model.kind.startswith("yield_"):
        from .yield_trend import draw_yield_trend
        if fitted is None:
            raise MissingDataError("yield model requires an available macro-state fit")
        return draw_yield_trend(fitted, months, n, rng)
    if model.kind == "dynamic":
        from .monthly_dynamic import fit_dynamic, draw_dynamic
        return draw_dynamic(fitted if fitted is not None else fit_dynamic(train), months, n, rng)
    if model.kind == "bootstrap":
        segments = year_segments(train.month_id.to_numpy())
        starts = valid_block_starts(segments, model.block_months)
        if not len(starts):
            raise MissingDataError("no complete contiguous bootstrap block")
        picks = rng.choice(starts, (n, -(-months // model.block_months)))
        indices = (picks[:, :, None] + np.arange(model.block_months)).reshape(n, -1)[:, :months]
        paths = x[indices]
    elif model.kind == "niw":
        fitted = B.fit_niw(x, prior)
        mu, cov = B.draw_niw_params(fitted, n, rng)
        paths = B.simulate_from_params(mu, cov, months, rng, np.arange(n))
    else:
        paths = prior.mu0 + rng.standard_normal((n, months, 4)) @ np.linalg.cholesky(prior.C0).T
    if not np.isfinite(paths).all():
        raise NumericalModelError("monthly predictive paths are not finite")
    return paths


def run_experiment(d: pd.DataFrame, models: list[MonthlyModel], prior: B.NIWPrior,
                   horizons: list[int], levels: list[float], seeds: list[int],
                   n_draws: int, as_of: pd.Timestamp, progress=None, macro_history=None):
    if (not horizons or any(h < 1 for h in horizons) or len(set(horizons)) != len(horizons)
            or not levels or any(not 0 < l < 1 for l in levels)
            or not seeds or len(set(seeds)) != len(seeds) or n_draws < 100
            or not models or len({m.name for m in models}) != len(models)):
        raise ValueError("invalid experiment controls")
    records, skipped, origin_audit = [], [], []
    origins = pd.date_range(d.date.min().to_period("M").start_time, as_of, freq="MS")
    for mi, model in enumerate(models):
        completed = 0
        for origin in origins:
            train = training_view(d, origin)
            if len(train) < model.training_months:
                continue
            targets, target_availability = {}, {}
            for h in sorted(horizons):
                y, reason = observed_target(d, origin, h, as_of)
                if y is None:
                    skipped.append({"method": model.name, "origin": str(origin.date()),
                                    "horizon_months": h, "reason": reason})
                else:
                    targets[h] = y
                    end = origin + pd.DateOffset(months=h) - pd.offsets.Day(1)
                    # Include the CPI anchor's release as well as every return month.
                    target_availability[h] = str(d.loc[(d.date >= origin - pd.offsets.Day(1))
                        & (d.date <= end), "available_date"].max().date())
            if not targets:
                continue
            last = train.iloc[-1]
            gap = int(origin.year * 12 + origin.month - last.month_id - 1)
            if gap < 0 or not (train.joint_available_date < origin).all():
                raise DataValidationError("forecast training crossed origin cutoff")
            origin_audit.append({"method": model.name, "origin": str(origin.date()),
                "training_months": len(train), "training_end": str(last.date.date()),
                "latest_training_availability": str(train.joint_available_date.max().date()),
                "unobserved_gap_months": gap})
            fitted = None
            if model.kind == "dynamic":
                from .monthly_dynamic import fit_dynamic
                fitted = fit_dynamic(train)
                origin_audit[-1].update(ar_cash_phi=fitted.phi[2], ar_inflation_phi=fitted.phi[3],
                    garch_omega=fitted.omega, garch_alpha=fitted.alpha, garch_beta=fitted.beta,
                    student_t_df=fitted.nu, garch_converged=True)
            if model.kind.startswith("yield_"):
                from .yield_trend import fit_yield_trend
                if macro_history is None:
                    raise MissingDataError("yield forecast requires macro history")
                equity_train = d.loc[(d.date < origin) & (d.available_date < origin)].copy()
                fitted = fit_yield_trend(equity_train, macro_history, origin, model.kind.removeprefix("yield_"))
                origin_audit[-1].update(macro_training_months=fitted.macro_rows,
                    adjacent_macro_training_pairs=fitted.macro_pairs, equity_training_months=len(equity_train),
                    first_macro_date=fitted.first_macro_date, last_macro_date=fitted.last_macro_date,
                    latest_macro_availability=fitted.latest_macro_availability,
                    inflation_trend=fitted.trend, inflation_state_variance=fitted.transition_variance,
                    inflation_observation_variance=fitted.observation_variance,
                    inflation_volatility_step=fitted.volatility_step)
            for seed in seeds:
                rng = np.random.default_rng(np.random.SeedSequence([seed, mi, origin.year * 12 + origin.month]))
                paths = predictive_paths(train, model, prior, gap + max(targets), n_draws, rng, fitted)
                for h, actual in targets.items():
                    pred = paths[:, gap:gap + h].sum(axis=1)
                    for j, series in enumerate(SERIES):
                        crps = crps_samples(pred[:, j], actual[j])
                        for level in levels:
                            alpha = 1 - level
                            lo, hi = np.quantile(pred[:, j], [alpha / 2, 1 - alpha / 2])
                            winkler = hi - lo + 2 / alpha * max(lo - actual[j], actual[j] - hi, 0)
                            records.append({"method": model.name, "seed": seed,
                                "origin": str(origin.date()), "horizon_months": h, "series": series,
                                "outcome_available_date": target_availability[h],
                                "level": level, "actual_log_growth": float(actual[j]),
                                "lower_log_growth": float(lo), "upper_log_growth": float(hi),
                                "hit": float(lo <= actual[j] <= hi), "width": float(hi - lo),
                                "winkler": float(winkler), "crps": crps})
            completed += 1
            if progress and completed % 30 == 0:
                progress(f"{model.name}: {completed} historical starting dates completed")
        if progress:
            progress(f"{model.name}: finished {completed} distinct starting dates (seeds do not add dates)")
    if not records:
        raise MissingDataError("no scorable monthly origins")
    return pd.DataFrame(records), pd.DataFrame(origin_audit), pd.DataFrame(skipped)


def block_mean_sensitivity(values: np.ndarray, dates: pd.Series, block: int,
                          reps: int, rng: np.random.Generator) -> tuple[float | None, float | None]:
    """Moving-block percentile range conditional on observed origin outcomes.

    Never bridge missing origin dates. This is a sensitivity interval, not a
    guaranteed coverage confidence interval in this short, nonstationary sample.
    """
    values = np.asarray(values, float)
    if not np.isfinite(values).all() or block < 1 or reps < 1 or not len(values):
        raise ValueError("invalid block sensitivity input")
    periods = pd.DatetimeIndex(dates).to_period("M").asi8
    if len(periods) != len(values) or np.any(np.diff(periods) <= 0):
        raise ValueError("origin dates must be unique, ordered and aligned")
    starts = valid_block_starts(year_segments(periods), block)
    if len(starts) < 2:
        return None, None
    picks = rng.choice(starts, size=(reps, -(-len(values) // block)))
    indices = (picks[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :len(values)]
    lo, hi = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return float(lo), float(hi)


def summarize(records: pd.DataFrame, reps: int, seed: int):
    keys = ["method", "horizon_months", "series", "level"]
    # Collapse simulation seeds at each real date before counting historical tests.
    by_origin = records.groupby(keys + ["origin"], as_index=False)[["hit", "width", "winkler", "crps"]].mean()
    rows, dependence, nonoverlap = [], [], []
    rng = np.random.default_rng(seed)
    for key, g in by_origin.groupby(keys, sort=True):
        method, h, series, level = key
        g = g.sort_values("origin")
        dates = pd.to_datetime(g.origin)
        raw = records.loc[(records.method == method) & (records.horizon_months == h)
                          & (records.series == series) & (records.level == level)]
        seed_coverage = raw.groupby("seed").hit.mean()
        row = dict(zip(keys, key))
        row.update(n_origins=len(g), coverage=float(g.hit.mean()), mean_width=float(g.width.mean()),
            mean_winkler=float(g.winkler.mean()), mean_crps=float(g.crps.mean()),
            coverage_seed_min=float(seed_coverage.min()), coverage_seed_max=float(seed_coverage.max()),
            nonoverlap_count_heuristic=len(g) / h,
            first_origin=g.origin.iloc[0], last_origin=g.origin.iloc[-1])
        rows.append(row)
        for block in sorted({max(1, h // 2), h, 2 * h}):
            low, high = block_mean_sensitivity(g.hit.to_numpy(), dates, block, reps, rng)
            dependence.append({**dict(zip(keys, key)), "origin_block_months": block,
                               "coverage_low": low, "coverage_high": high,
                               "status": "SENSITIVITY_ONLY" if low is not None else "TOO_FEW_CONTIGUOUS_ORIGINS"})
        periods = pd.DatetimeIndex(dates).to_period("M").asi8
        # Each offset uses disjoint target periods. Offsets are alternatives, not replicates.
        for offset in range(h):
            subset = g.loc[periods % h == offset]
            if len(subset):
                nonoverlap.append({**dict(zip(keys, key)), "offset": offset,
                    "n_origins": len(subset), "coverage": float(subset.hit.mean())})
    # Fair model comparison uses the intersection of actual scored dates.
    common_rows = []
    for (h, series, level), g in by_origin.groupby(["horizon_months", "series", "level"]):
        expected = records.method.nunique()
        counts = g.groupby("origin").method.nunique()
        common = set(counts.index[counts == expected])
        for method, sub in g.loc[g.origin.isin(common)].groupby("method"):
            common_rows.append({"method": method, "horizon_months": h, "series": series, "level": level,
                                "n_common_origins": len(sub), "coverage": float(sub.hit.mean()),
                                "mean_winkler": float(sub.winkler.mean()), "mean_crps": float(sub.crps.mean())})
    return pd.DataFrame(rows), pd.DataFrame(dependence), pd.DataFrame(nonoverlap), pd.DataFrame(common_rows)
