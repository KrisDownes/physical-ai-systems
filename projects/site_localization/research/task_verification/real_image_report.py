"""Evaluate fixed predictions against previously frozen interval annotations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main(root):
    freeze = json.loads((root/'pre_execution_freeze.json').read_text())
    for name, digest in freeze['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    labels = {r['index']: r for r in json.loads((root/'reference_labels.json').read_text())}
    predictions = json.loads((root/'predictions.json').read_text())
    rows = []
    for p in predictions:
        t = labels[p['index']]
        row = {k: t[k] for k in ['index', 'file', 'gauge_type', 'input_condition', 'reference_status']}
        row.update(execution=p['execution'], decision=None, prediction=None,
                   reference_low=None, reference_high=None, min_error_units=None,
                   max_error_units=None, min_error_fullscale=None, max_error_fullscale=None,
                   judgment='source_unavailable', peak_support=None, competing_support=None)
        if t['reference_interval'] is not None:
            row['reference_low'], row['reference_high'] = t['reference_interval']
        if p['execution'] == 'complete':
            pred = p['prediction']; value = pred.get('reading_value')
            row.update(decision=pred['decision'], prediction=value,
                       peak_support=pred.get('peak_support'), competing_support=pred.get('competing_support'))
            row['judgment'] = 'abstention' if pred['decision'] != 'accept' else 'accepted_reference_indeterminate'
            if value is not None and t['reference_status'] == 'determinate':
                lo, hi = t['reference_interval']; span = t['scale_span']
                emin = max(lo-value, 0, value-hi); emax = max(abs(value-lo), abs(value-hi))
                row.update(min_error_units=emin, max_error_units=emax,
                           min_error_fullscale=emin/span, max_error_fullscale=emax/span)
                if pred['decision'] == 'accept':
                    row['judgment'] = ('accepted_certainly_correct' if emax/span <= .02
                                       else 'accepted_certainly_incorrect' if emin/span > .02
                                       else 'accepted_numerically_ambiguous')
        rows.append(row)

    def summary(subset):
        usable = [r for r in subset if r['execution'] == 'complete']
        return {'planned_source_cases': len(subset), 'reader_completed': len(usable),
                'source_unavailable': sum(r['execution'] == 'source_unavailable' for r in subset),
                'accepted': sum(r['decision'] == 'accept' for r in usable),
                'abstained': sum(r['decision'] == 'abstain' for r in usable),
                'reference_determinate': sum(r['reference_status'] == 'determinate' for r in usable),
                'certainly_correct_accepted': sum(r['judgment'] == 'accepted_certainly_correct' for r in usable),
                'certainly_incorrect_accepted': sum(r['judgment'] == 'accepted_certainly_incorrect' for r in usable),
                'ambiguous_accepted': sum(r['judgment'] == 'accepted_numerically_ambiguous' for r in usable),
                'indeterminate_reference_accepted': sum(r['judgment'] == 'accepted_reference_indeterminate' for r in usable),
                'coverage_of_usable_images': sum(r['decision'] == 'accept' for r in usable)/len(usable) if usable else None,
                'coverage_of_planned_sources': sum(r['decision'] == 'accept' for r in usable)/len(subset) if subset else None}
    result = {'scope': 'Reading/readability kernel only; manually supplied per-image geometry, one visual-model annotator, no field reference',
              'all': summary(rows), 'supported_gauge_types': {
                  typ: summary([r for r in rows if r['gauge_type'] == typ])
                  for typ in ['A_0_1.6_kPa', 'B_0_25_MPa']},
              'unsupported_gauge_types': {'cases': 0, 'coverage': None, 'error_rate': None,
                                          'reason': 'No scored target of a clearly unsupported gauge class in this subset.'},
              'runtime_failures': sum(p['execution'] == 'runtime_failure' for p in predictions),
              'reader_calls': sum(p['reader_called'] for p in predictions),
              'reader_kernel_wall_s': sum(p.get('wall_s', 0) for p in predictions),
              'reader_kernel_cpu_s': sum(p.get('cpu_s', 0) for p in predictions),
              'tolerance_fullscale_fraction': .02,
              'incorrect_acceptance_rate_identifiable': False,
              'uncertainty': 'No confidence interval: source episode/asset identity unknown, correlated photographs, single annotator and interval ambiguity.',
              'no_retuning': True}
    with (root/'outcomes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    main(p.parse_args().root)
