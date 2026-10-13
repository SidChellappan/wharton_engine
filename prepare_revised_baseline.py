"""Freeze one reasoned revision before fresh simulation; preserve packet originals."""
from pathlib import Path
import copy
import hashlib
import json
import yaml

root = Path(__file__).resolve().parent
cfgdir = root / 'configs'
candidates = yaml.safe_load((cfgdir / 'candidates_team.yaml').read_text())
revised = copy.deepcopy(candidates['families'][0])
revised.update(id='revised_baseline', role='revised_baseline',
    description='Earlier de-risking to improve prolonged-downturn resilience; fixed before fresh validation.',
    provenance='User requested revision after original results; allocations frozen before new draws.')
revised['weights_path'] = {year: {'equity': equity, 'treasury': round(.95-equity, 2), 'cash': .05}
                          for year, equity in zip(range(2027,2033), [.40,.40,.30,.25,.15,.10])}
candidates['families'].append(revised)
candidates['benchmarks'].append('revised_baseline')
candidates['always_validate'].append('revised_baseline')
candidates['version'] = 'team-revision-1.0'
(cfgdir / 'candidates_revised_baseline.yaml').write_text(yaml.safe_dump(candidates, sort_keys=False))
for name, source in [('history','team_history_diagnostic.yaml'), ('fallback','team_run.yaml')]:
    cfg = yaml.safe_load((cfgdir/source).read_text())
    cfg['run']['name'] = 'revised_baseline_' + name
    cfg['run']['master_seed'] = 2026101102
    cfg['candidates_file'] = 'candidates_revised_baseline.yaml'
    cfg['conditional_2031']['frozen_candidate'] = 'revised_baseline'
    # Conservative all-portfolio fee covers IVV/IEF weighted fund costs through the glide.
    cfg['costs']['annual_fee_rate'] = .0015
    cfg['costs']['provenance'] = 'Other packet-run frictions unchanged; annual fee raised from 5bp to 15bp to cover indicative IVV/IEF fund expenses. Thresholds unchanged.'
    (cfgdir / ('revised_baseline_'+name+'.yaml')).write_text(yaml.safe_dump(cfg,sort_keys=False))
manifest = {'candidate': revised, 'master_seed': 2026101102,
            'policy_sha256': hashlib.sha256((cfgdir/'risk_policy_team.yaml').read_bytes()).hexdigest(),
            'validation_paths':50000, 'paths_per_mandatory_stress':50000,
            'selection':'Single reasoned revision; no search or selection using fresh validation outcomes.',
            'cost_change':'Annual portfolio fee 0.05% -> 0.15%; all original candidates rerun on same new assumptions.'}
(root/'revised_baseline_preregistration.json').write_text(json.dumps(manifest,indent=2))
print('Frozen revised baseline and two runs; unchanged policy hash:', manifest['policy_sha256'])
