"""Package completed real-history runs and report every rule without rerunning them."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import numpy as np
import pandas as pd
import yaml
from wharton_engine.io import source_tree_hash


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    dest = repo / 'docs/model_repair_2026-10-11'
    labels = ['historical_diagnostic', 'fallback', 'annual_bootstrap']
    summaries, rules, coverages, outcomes, verification = [], [], [], [], []
    manifests, runs = {}, {}
    for label in labels:
        run = sorted(p for p in args.run_root.glob('model_repair_'+label+'_*') if (p/'run_manifest.json').exists())[-1]
        manifest = json.loads((run/'run_manifest.json').read_text())
        assert manifest['as_of'] == '2026-10-11'
        assert manifest['code_revision']['local_changes'] == []
        assert manifest['code_revision']['source_tree_sha256_16'] == source_tree_hash(repo/'src/wharton_engine')
        assert manifest['path_counts']['validation'] == manifest['path_counts']['stress'] == 50000
        for item in manifest['inputs'].values():
            if isinstance(item, dict) and 'path' in item and 'sha256' in item:
                assert digest(Path(item['path'])) == item['sha256']
        target = dest/'models'/label
        target.mkdir(parents=True, exist_ok=True)
        for file in run.iterdir():
            if file.is_file():
                shutil.copy2(file, target/file.name)
        gates = json.loads((run/'gate_results.json').read_text())['validation']
        scores = pd.read_csv(run/'risk_scorecards.csv').set_index('candidate_id')
        stresses = pd.read_csv(run/'stress_results.csv')
        assert len(gates) == 8 and not any(g['status']=='PASS' for g in gates.values())
        for candidate, gate in gates.items():
            row = scores.loc[candidate]
            stress_rows = stresses[stresses.candidate_id == candidate]
            substantive = [r for r in gate['rules'] if r['rule_id'] not in ['forecast_evidence', 'issuer_concentration']]
            summaries.append(dict(model=label, candidate=candidate, status=gate['status'],
                other_rules_all_pass=all(r['status']=='PASS' for r in substantive),
                payment_success=row.p_all_payments, payment_lower_bound=row.p_all_payments_lower_one_sided,
                reserve_success=row.p_full_reserve, es_unpaid=row.es_unpaid,
                p95_drawdown=row.p95_max_drawdown, forced_sales=row.p_payment_requires_sale,
                facility_median=row.facility_p50, minimum_stress_success=stress_rows.p_all_payments.min(),
                minimum_stress_lower_bound=stress_rows.p_all_payments_lower_one_sided.min(),
                failed_rules='; '.join(r['rule_id'] for r in gate['rules'] if r['status']=='FAIL'),
                missing_rules='; '.join(r['rule_id'] for r in gate['rules'] if r['status'] not in ['PASS','FAIL'])))
            for rule in gate['rules']:
                rules.append(dict(model=label, candidate=candidate, overall_status=gate['status'], **rule))
        forecast = json.loads((run/'forecast_validation.json').read_text())
        diagnostics = forecast['diagnostics']
        validated = diagnostics['validated_method']
        for horizon, key in [(1, 'rolling_origin'), (5, 'horizon_validation')]:
            metric = diagnostics.get(key, {})
            if horizon == 1:
                metric = metric.get('niw', metric.get('block_bootstrap_baseline', metric))
            for level, series in metric.get('coverage', {}).items():
                if series is None:
                    continue
                for name, value in series.items():
                    minimum = float(level)-diagnostics['model_gate_thresholds']['coverage_tolerance'][level]
                    coverages.append(dict(run=label, method=validated, selected_method=forecast['method'], horizon_years=horizon,
                        interval=float(level), series=name, coverage=value, minimum_coverage=minimum,
                        passes=abs(value-float(level)) <= diagnostics['model_gate_thresholds']['coverage_tolerance'][level]))
        for reason in diagnostics.get('reasons', []):
            outcomes.append(dict(experiment=label, method=validated, status=diagnostics['status'], reason=reason))
        checked = json.loads((run/'report_verification.json').read_text())
        assert checked['ok'] and checked['n_problems'] == 0
        verification.append(checked)
        manifests[label], runs[label] = manifest, run.name
    summary = pd.DataFrame(summaries)
    assert len({m['code_revision']['git'] for m in manifests.values()}) == 1
    assert summary[summary.candidate.isin(['earlier_derisking', 'bill_heavy_control'])].other_rules_all_pass.all()
    summary.to_csv(dest/'repair_results.csv', index=False)
    pd.DataFrame(rules).to_csv(dest/'all_rule_results.csv', index=False)
    coverage = pd.DataFrame(coverages)
    coverage.to_csv(dest/'full_run_forecast_coverage.csv', index=False)
    for experiment in ['original','cash_comparison']:
        audit = json.loads((dest/f'forecast_audit_{experiment}.json').read_text())
        assert not any(v['status']=='PASS' for v in audit['methods'].values())
        for method, diagnostics in audit['methods'].items():
            for reason in diagnostics['reasons']:
                outcomes.append(dict(experiment='standalone_'+experiment, method=method, status=diagnostics['status'], reason=reason))
    pd.DataFrame(outcomes).to_csv(dest/'forecast_failure_reasons.csv', index=False)
    original = pd.read_csv(repo/'data/public_20261011/annual_returns.csv')
    comparison = pd.read_csv(repo/'data/public_cash_comparison_20261011/annual_returns.csv')
    assert original.drop(columns='cash_tr').equals(comparison.drop(columns='cash_tr'))
    monthly = pd.read_csv(repo/'data/public_cash_comparison_20261011/observed_monthly_bill_returns.csv')
    assert len(monthly) == 1164
    histlog = np.log1p(original[['cash_tr','inflation']])
    correlations = {key: float(histlog[key].autocorr()) for key in histlog}
    causes = pd.read_csv(dest/'cause_attribution.csv')
    assert len(causes) == 24
    for col in ['failures_equal_initial_reserve_gap','facility_zero_on_all_failed_paths', 'retaining_all_surplus_does_not_change_failures']:
        assert causes[col].all()
    assert causes.reproduction_error.max() < 1e-12
    stress_audit = pd.read_csv(dest/'stress_accounting_checks.csv')
    assert stress_audit.max_dollar_difference.max() < 1e-6
    scalar_count = sum(len(pd.read_csv(args.run_root/runs[label]/'audit_reference_vs_vectorized.csv')) for label in labels)
    test_count = int(re.search(r'(\d+) tests collected', (dest/'test_collection.txt').read_text()).group(1))
    # Double quiet plus disabled warnings suppresses pytest's final summary.
    # The complete stdout contains one passing dot per collected test; the
    # completed process exit code was 0 and is recorded in test_execution.json.
    assert (dest/'tests.txt').read_text().count('.') == test_count
    test_evidence = json.loads((dest/'test_execution.json').read_text())
    assert test_evidence['exit_code'] == 0
    assert test_evidence['source_tree_sha256_16'] == source_tree_hash(repo/'src/wharton_engine')
    qa = dict(source_commit=manifests[labels[0]]['code_revision']['git'], run_ids=runs, tests_passed=test_count,
        source_tree_sha256_16=source_tree_hash(repo/'src/wharton_engine'),
        unchanged_input_hash_checks=True, original_non_cash_columns_unchanged=True,
        monthly_cash_observations=len(monthly), lag_one_log_correlations=correlations,
        full_rule_records=len(rules), full_run_scalar_path_checks=scalar_count,
        additional_stress_scalar_path_checks=len(stress_audit),
        stress_scalar_max_dollar_difference=float(stress_audit.max_dollar_difference.max()),
        st2_attribution_matches_all_24_candidate_model_results=True,
        figures_checked=sum(v['n_figures_checked'] for v in verification), figure_registry_problems=0,
        two_new_allocations_all_other_rules_pass=True, overall_passes=0,
        all_five_forecast_methods_without_pass_on_both_sources=True)
    (dest/'evidence_verification.json').write_text(json.dumps(qa,indent=2), encoding='utf-8')

    def pct(value):
        return f'{value:.3%}'

    rows = ['| Forecast scenario | Allocation | All payments | Worst stress | Worst stress lower bound | Other 12 rules | Overall |',
            '|---|---|---:|---:|---:|---|---|']
    for row in summary.itertuples():
        if row.candidate not in ['protective','baseline','higher_equity','earlier_derisking','bill_heavy_control']:
            continue
        rows.append(f'| {row.model} | {row.candidate} | {pct(row.payment_success)} | {pct(row.minimum_stress_success)} | {pct(row.minimum_stress_lower_bound)} | {"PASS" if row.other_rules_all_pass else "See rule CSV"} | {row.status} |')
    failure_rows=['| Method | Horizon (years) | Series | Interval | Coverage | Minimum |', '|---|---:|---|---:|---:|---:|']
    for row in coverage[~coverage.passes].drop_duplicates(['method','horizon_years','interval','series','coverage']).itertuples():
        failure_rows.append(f'| {row.method} | {row.horizon_years} | {row.series} | {row.interval:.0%} | {pct(row.coverage)} | {row.minimum_coverage:.0%} |')
    baseline = causes[(causes.model=='historical_diagnostic') & (causes.candidate=='baseline')].iloc[0]
    primary = json.loads((dest/'models/historical_diagnostic/forecast_validation.json').read_text())
    content = f'''# Model failure audit and repair — October 11, 2026

Both excessive risk before the payment reserve is established and inadequate forecast evidence explain the failures. The software now runs all three complete experiments successfully, but **no portfolio has overall PASS and no forecast method is validated on either tested data definition**. Two conservative allocation trials pass the 12 applicable rules other than forecast evidence and security concentration under all three scenario generators. They are conditional review candidates, not a competition execution portfolio.

## Concrete breaking point

The payment reserve must be purchased in 2033. Its all-in planning cost is **$404,907.44** under the unchanged October 9, 2026 Treasury snapshot assumption. In the real-history NIW diagnostic, baseline ST2 payment success is **{pct(baseline.st2_success)}**, below the 90% stress requirement. All **{int(baseline.n_failed):,}** failed paths have an initial reserve funding gap, averaging **${baseline.mean_initial_gap_on_failures:,.2f}** among failures.

This result is reproduced for all 24 candidate/model ST2 combinations in [cause_attribution.csv](cause_attribution.csv). Every failed path already sets the facility allocation to zero. Changing retention to 100% leaves the failure cohort identical. Thus facility spending or insufficient post-reserve retention is not the cause of these ST2 failures. For combinations with zero failures, these cohort statements are vacuous; they are not additional evidence about failed paths.

Removing all implementation costs improves baseline NIW ST2 success only from {pct(baseline.st2_success)} to {pct(baseline.zero_cost_counterfactual_success)}. It remains below 90%; costs contribute but do not account for the shortfall. This is an attribution counterfactual, not permission to remove costs from the policy.

**Risk repair:** reduce equity earlier and keep more liquid capital before 2033. The explicit earlier-derisking trial below clears all numeric gates in each generator. Keeping more surplus after 2033 alone is ineffective. Lower risk reduces prospective facility resources; median facility outcomes are retained in [repair_results.csv](repair_results.csv).

## Forecast failure and data evidence

The existing NIW validator found real failures. It was incomplete for other methods and for the long planning horizon. Annual log cash lag-one correlation is **{correlations['cash_tr']:.3f}** and inflation is **{correlations['inflation']:.3f}**. Independent annual draws omit this persistence; chronological five-year checks expose severe undercoverage. This is supported model misspecification, not merely a failed numerical sampler. Persistence is a concrete candidate mechanism, not proof that it explains every calibration error.

The table reports every failed interval coverage check from the final full runs, using the exact fit-diagnostics stream. The fallback run preserves the failed primary NIW diagnostics: those checks belong to NIW, not the selected assumption generator. Duplicate NIW results are shown once. NIW's posterior predictive worst-three-year equity check also fails: p=0.994 exceeds 0.99. The 80% interval requires coverage at least 65%; the 95% interval requires at least 85%. Existing tolerances were retained. The repair experiment additionally requires those checks at five years for all four series, with at least 10 n/H effective origins, frozen before comparison. Overlapping-window n/H is a dependence heuristic, not a formal independent sample count.

{chr(10).join(failure_rows)}

The separate [five-method original-data audit](forecast_audit_original.json) and [cash-source comparison audit](forecast_audit_cash_comparison.json) reach the same qualitative result:

- NIW and fitted fixed covariance fail held-out coverage.
- Monthly bootstrap lacks joint monthly equity/Treasury/cash/CPI data.
- Annual bootstrap fits actual annual observations, but fails held-out one-year and five-year coverage. Five-year observed blocks do not establish a calibrated regime model.
- Stated assumptions remain conditional scenarios with insufficient forecast evidence.

The standalone command uses separate deterministic validation streams, so Monte Carlo bootstrap interval results can differ slightly from the full-run table. Both outputs remain failures. [forecast_failure_reasons.csv](forecast_failure_reasons.csv) preserves each exact reason and its experiment.

The original cash input is an average three-month bill-rate proxy, following the [NYU historical table convention](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html). A separate comparison compounds 1,164 real monthly RF observations into 97 annual returns for 1928–2024, from the [Kenneth French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/f-f_factors.html). The French definition uses one-month bills, with a provider transition in June 2024. It is a different observed cash-return definition, not an exact reconstruction of the original three-month proxy. Equity, Treasury, inflation, years and assigned annual availability are identical. No synthetic historical months are created; cash-only monthly data does not enable joint monthly bootstrap. The comparison does **not** cure forecast calibration. Raw download, hashes, provider definitions and compound calculations are preserved in [the comparison source manifest](../../data/public_cash_comparison_20261011/SOURCE_MANIFEST.yaml).

## Fixes delivered

- Added method-specific fixed-covariance and annual-bootstrap validators, an all-method audit command and chronological multi-year coverage. Bootstrap fallback is evaluated using its own distribution, not NIW evidence.
- Missing or nonfinite coverage, missing predictive density, absent required posterior checks, inadequate scored origins and invalid validator controls cannot produce PASS.
- Aligned monthly-bootstrap held-out history requirements with the fitter. Bootstrap training history respects training-start limits.
- Reject malformed supplied availability dates; both CPI levels must be published before deriving inflation. Missing availability remains an explicit release-lag assumption, not a historical publication vintage. Empty histories return an explicit data outcome.
- Added an optional annual-block generator using contiguous joint observed years, without inventing monthly data. It is retained as an unsuccessful alternative because its validator fails. It is not silently promoted into old configurations.
- Gaussian sampling is checked against its own analytical distribution; multi-year mean uncertainty is held across years consistently. Non-NIW 2031 beliefs are explicitly frozen rather than presented as conditional posterior updates.
- Report headings identify the actual diagnostic method separately from the selected fallback, and name Gaussian versus Student-t predictive distributions correctly.

Validator usage, scope and remaining assumptions are documented in [forecast_validation.md](../forecast_validation.md). The software fixes do not change observed history, original six allocations, or portfolio risk thresholds.

## Fresh financial comparison

Each generator uses 5,000 development paths, **50,000 fresh validation paths and 50,000 paths for each of the four stresses**, master seed **2026101103**. NIW and annual bootstrap are explicitly retained despite failed validation for diagnosis; assumption output is conditional. Probabilities and their Monte Carlo bounds quantify simulation uncertainty under each named generator, not verified real-world probabilities.

{chr(10).join(rows)}

All 24 configurations remain NEEDS_MORE_EVIDENCE or FAIL overall. Complete results for the six original plans, including benchmark and retention variants, appear in [repair_results.csv](repair_results.csv). Every one of the **{len(rules)} validation-rule outcomes**, threshold and reason appears in [all_rule_results.csv](all_rule_results.csv). Base thresholds remain: 95% payment success and full reserve, worst-5% mean unpaid ≤$50,000, 95th-percentile annual-endpoint drawdown ≤30%, forced sales ≤5%, equity ≤65%, each stress success ≥90% supported by its 95% one-sided lower bound, retention ≥25% and the stated $50,000 retained floor when surplus permits. Simulated issuer/sector checks remain unevaluated without actual holdings.

### Two exploratory allocations

These are new diagnostic trials, separate from the original workbook instructions. Neither was selected from an optimization claiming validated future returns.

| Year | Earlier equity | Earlier Treasury | Earlier cash | Bill-control equity | Bill-control Treasury | Bill-control cash |
|---|---:|---:|---:|---:|---:|---:|
| 2027 | 20% | 30% | 50% | 0% | 5% | 95% |
| 2028 | 20% | 25% | 55% | 0% | 5% | 95% |
| 2029 | 10% | 25% | 65% | 0% | 5% | 95% |
| 2030 | 5% | 25% | 70% | 0% | 5% | 95% |
| 2031 | 0% | 20% | 80% | 0% | 5% | 95% |
| 2032 | 0% | 10% | 90% | 0% | 5% | 95% |

Both retain 40% of post-reserve surplus, apply the $50,000 retained floor when feasible, and use 25% equity/70% Treasury/5% cash for retained capital after reserve establishment. This shows that pre-2033 risk reduction addresses the simulated financing failure. It does not demonstrate that a bill-heavy portfolio satisfies competition implementation requirements or an unvalidated forecast.

## What is still missing

1. **Validated forecast:** none found. A persistent rate/inflation or regime model requires fitting, chronological calibration and a newly frozen evaluation before approval; widening intervals merely until old failures disappear would overfit the audit. The annual-block attempt failed and is not called fixed.
2. **Actual security holdings:** tickers, weights, issuer and sector mapping are needed for the 5% single-security and 20% sector limits. Broad asset-class weights cannot pass these checks.
3. **Point-in-time evidence:** the 97-year annual panel is a revised historical extract, with assumed January publication availability. Real historical vintages are not reconstructed. Available history ends in 2024; later annual observations were not invented.
4. **Implementation and finer risk data:** joint monthly Treasury total returns appropriate to the holdings, real instrument prices, bond maturity/face/quote data and a dated funding curve. Annual-endpoint drawdown omits within-year losses. Today's curve held unchanged in 2033 is a scenario assumption, not a future quote.
5. **Team/advisor decision:** policy approval is false in the input packet. Even approval would not repair the failed forecast or absent holdings evidence.

No competition execution portfolio is issued because the required complete PASS is absent. The two concrete allocations and all evidence are ready for review; no holdings, approvals or observed market data were fabricated.

## Verification and provenance

The final source revision is **{qa['source_commit']}**, source-tree hash **{qa['source_tree_sha256_16']}**, clean during all three runs. The unchanged workbook SHA256 is `57b11b7f9eb36bb599958c906df24280c8a3600bbce974301e2133ef5f9ff548`. Input file hashes and seed streams are retained in each manifest; the Treasury plan/policy hash remains `898434d460c74e9f`.

**{test_count} tests passed** ([test evidence](tests.txt)); full-run independent scalar checks cover {scalar_count} paths. Additional stress attribution independently checks {len(stress_audit)} paths, with maximum scalar-versus-vectorized dollar difference {qa['stress_scalar_max_dollar_difference']:.3g}, below the $0.000001 tolerance. The report registry checks {qa['figures_checked']:,} figures with zero registered problems. This checks numerical/registry consistency; it is not a claim of manual visual inspection of every chart.

{chr(10).join('- '+label+': `'+runs[label]+'` — [review packet](models/'+label+'/review_packet.html), [manifest](models/'+label+'/run_manifest.json)' for label in labels)}

Small complete report artifacts are committed. Large path arrays and sampled event ledgers remain local and can be regenerated. [evidence_verification.json](evidence_verification.json) records package checks; [frozen_experiment.json](frozen_experiment.json) records the allocation/configuration freeze.

From the repository root after installing the package:

```text
python tools/prepare_model_repair_runs.py
python -m wharton_engine.cli audit-forecasts --config docs/model_repair_2026-10-11/configs/historical_diagnostic.yaml --out original_audit.json
python -m wharton_engine.cli audit-forecasts --config docs/model_repair_2026-10-11/configs/cash_source_comparison.yaml --out cash_audit.json
python -m wharton_engine.cli run --config docs/model_repair_2026-10-11/configs/historical_diagnostic.yaml
python -m wharton_engine.cli run --config docs/model_repair_2026-10-11/configs/fallback.yaml
python -m wharton_engine.cli run --config docs/model_repair_2026-10-11/configs/annual_bootstrap.yaml
```

Run files record the source commit used at execution. Later evidence-packaging commits leave that measured source-tree hash unchanged.
'''
    (dest/'README.md').write_text(content, encoding='utf-8')
    files = {str(p.relative_to(dest)).replace('\\','/'): digest(p) for p in dest.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'}
    (dest/'artifact_hashes.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    print(json.dumps(qa, indent=2))
    print(summary[['model','candidate','status','other_rules_all_pass','minimum_stress_success']].to_string(index=False))


if __name__ == '__main__':
    main()
