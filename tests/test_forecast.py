"""Acceptance test 10 plus model-check / fallback behaviour."""

import numpy as np
import pytest
from scipy import stats

from wharton_engine.config import PriorSpec
from wharton_engine.errors import NumericalModelError
from wharton_engine.forecast import bayesian as B
from wharton_engine.forecast.checks import analytic_predictive_check, crps_gaussian, crps_samples, select_forecast
from wharton_engine.scenarios.paths import generate_scenarios


def _prior(kappa0=2.0, nu0=8.0):
    spec = PriorSpec(mean_log={"equity": .06, "treasury": .03, "cash": .02, "inflation": .02},
                     vol_log={"equity": .16, "treasury": .06, "cash": .01, "inflation": .01},
                     correlation=[[1, -.1, 0, 0], [-.1, 1, .3, 0], [0, .3, 1, .4], [0, 0, .4, 1]],
                     kappa0=kappa0, nu0=nu0, provenance="test")
    return B.prior_from_spec(spec)


def test_posterior_equations_match_hand_computation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4)) * [0.15, 0.05, 0.01, 0.01] + [0.07, 0.03, 0.02, 0.02]
    pr = _prior()
    post = B.fit_niw(x, pr)
    n, xb = 30, x.mean(0)
    S = (x - xb).T @ (x - xb)
    assert post.kappa_n == pr.kappa0 + n and post.nu_n == pr.nu0 + n
    assert np.allclose(post.mu_n, (pr.kappa0 * pr.mu0 + n * xb) / (pr.kappa0 + n))
    d = (xb - pr.mu0)[:, None]
    assert np.allclose(post.psi_n, pr.psi0 + S + pr.kappa0 * n / (pr.kappa0 + n) * d @ d.T)
    # scale vs expected covariance: Psi0 = (nu0 - d - 1) C0
    assert np.allclose(pr.psi0 / (pr.nu0 - 5), pr.C0)
    np.linalg.cholesky(post.psi_n)


def test_sequential_update_equals_batch_fit():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(24, 4)) * 0.05
    pr = _prior()
    full = B.fit_niw(x, pr)
    seq = B.update_niw(B.fit_niw(x[:20], pr), x[20:])
    assert np.allclose(full.mu_n, seq.mu_n) and np.allclose(full.psi_n, seq.psi_n)
    assert full.kappa_n == seq.kappa_n and full.nu_n == seq.nu_n


def test_sampled_predictive_matches_student_t():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(35, 4)) * [0.16, 0.05, 0.01, 0.01]
    post = B.fit_niw(x, _prior())
    chk = analytic_predictive_check(post, np.random.default_rng(3), n=60000)
    assert chk["max_error_sd_units"] < 0.03
    # moments: sampled sd vs analytic t sd
    assert np.allclose(chk["predictive_sd_sampled"], chk["predictive_sd_analytic"], rtol=0.05)


def test_joint_dependence_and_positive_definite_draws():
    rng = np.random.default_rng(4)
    L = np.linalg.cholesky(np.array([[1, .7, 0, 0], [.7, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    x = (rng.normal(size=(200, 4)) @ L.T) * 0.1
    post = B.fit_niw(x, _prior(kappa0=0.5))
    mu, sig = B.draw_niw_params(post, 500, np.random.default_rng(5))
    assert np.all(np.linalg.eigvalsh(sig) > 0)
    sims = B.simulate_from_params(mu, sig, 1, np.random.default_rng(6), np.arange(500))[:, 0, :]
    big = B.simulate_from_params(mu[:1], sig[:1], 20000, np.random.default_rng(7), np.zeros(1, int))[0]
    assert np.corrcoef(big[:, 0], big[:, 1])[0, 1] > 0.5     # dependence is simulated jointly
    assert sims.shape == (500, 4)


def test_prior_validation():
    with pytest.raises(Exception):
        _prior(nu0=5.0)          # needs nu0 > d + 1
    with pytest.raises(Exception):
        _prior(kappa0=0.0)


def test_crps_helpers():
    rng = np.random.default_rng(0)
    s = rng.normal(0.1, 0.2, 200000)
    assert crps_samples(s, 0.3) == pytest.approx(crps_gaussian(0.1, 0.2, 0.3), rel=0.01)


def test_demo_selection_reports_no_validated_forecast(demo_selection, niw_snapshot):
    """Readiness plan s.1: NIW undercovers persistent cash; the BVAR candidate improves but is still
    gated out; no assumption fallback is promoted to PASS."""
    snap, rep = demo_selection
    assert rep["outcome"] == "NO_VALIDATED_FORECAST"
    assert snap.method == "assumption" and snap.status == "NEEDS_MORE_EVIDENCE"
    assert snap.fallback["status"] == "CONDITIONAL_SCENARIO_NOT_EVIDENCE"
    niw, bvar = rep["candidates"]["niw"], rep["candidates"]["bvar"]
    assert niw["status"] == "POOR_PERFORMANCE" and any("cash" in r for r in niw["reasons"])
    assert any("cash_lag1_autocorr" in r for r in niw["reasons"])
    c_niw = niw["rolling_origin"]["horizons"]["5"]["coverage"]["0.95"]["cash"]
    c_bvar = bvar["rolling_origin"]["horizons"]["5"]["coverage"]["0.95"]["cash"]
    assert c_bvar > c_niw + 0.2                      # persistence-aware model materially better ...
    assert bvar["status"] != "PASS"                  # ... but not validated on this fixture
    for d in (niw, bvar):                            # enough scored origins for the plan's counts
        assert d["rolling_origin"]["horizons"]["1"]["n_windows"] >= 10
        assert d["rolling_origin"]["horizons"]["5"]["effective_n"] >= 10
        assert "6" in d["rolling_origin"]["horizons"]  # six-year accumulation reported
    b = generate_scenarios(niw_snapshot, 500, 2027, 2041, 1, "development")
    assert b.returns.shape == (500, 15, 3) and np.all(b.returns > -1)
    assert b.diagnostics["clipping"].startswith("none")


def test_short_history_is_insufficient_evidence(demo_ds, demo_cfg):
    from dataclasses import replace
    short = replace(demo_ds, annual=demo_ds.annual.tail(40).reset_index(drop=True))
    snap, rep = select_forecast(short, demo_cfg.forecast, np.random.default_rng(0), [])
    for m in ("niw", "bvar"):
        st = rep["candidates"][m]
        assert st["status"] == "INSUFFICIENT_EVIDENCE"
        assert any("effective n" in r for r in st["reasons"])     # 5-year windows / 5 < 10
    assert snap.fallback["active"] and snap.status != "PASS"


def test_nonfinite_training_data_is_numerical_error():
    x = np.zeros((25, 4))
    x[3, 0] = np.nan
    with pytest.raises(NumericalModelError):
        B.fit_niw(x, _prior())
