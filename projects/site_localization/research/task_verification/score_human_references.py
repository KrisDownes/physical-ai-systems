"""Compare saved predictions to human-finalized intervals. Never run the reader."""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def main(packet):
    packet = packet.resolve()
    root = Path(__file__).resolve().parents[2]
    old = root/'results/real_gauge_feasibility_20261023'
    preserved = json.loads((packet/'preservation_before.json').read_text())
    assert all(hashlib.sha256((root/n).read_bytes()).hexdigest() == h for n,h in preserved.items())
    confirmation = json.loads((packet/'human_finalization_confirmation.json').read_text())
    assert hashlib.sha256((packet/'reconciled_review_draft.json').read_bytes()).hexdigest() == confirmation['draft_sha256']
    reference = json.loads((packet/'final_reference_record.json').read_text())
    assert reference['complete'] and reference['reference_decisions_recorded']
    predictions = {r['file']: r for r in json.loads((old/'predictions.json').read_text())}
    configs = {r['file']: r for r in json.loads((old/'method_inputs.json').read_text())}
    rows = []
    for r in reference['cases']:
        record = predictions[r['file']]
        assert record['execution'] == 'complete' and record['prediction']['decision'] == 'accept'
        config = configs[r['file']]
        span = Decimal(str(config['max_value'])) - Decimal(str(config['min_value']))
        tol = Decimal('.02')*span
        pred = Decimal(str(record['prediction']['reading_value']))
        row = {'id': r['id'], 'file': r['file'], 'units': r['units'], 'prediction': str(pred),
               'reference_low': r.get('low'), 'reference_high': r.get('high'),
               'tolerance_units': str(tol), 'minimum_error_units': None,
               'maximum_error_units': None, 'minimum_error_fullscale': None,
               'maximum_error_fullscale': None, 'classification': 'unresolved', 'reason': None}
        scale_matches = (r['units'] == config['units'] and Decimal(r['minimum']) == Decimal(str(config['min_value']))
                         and Decimal(r['maximum']) == Decimal(str(config['max_value'])))
        if r['reference_status'] != 'interval':
            row['reason'] = 'Human reference not determinable'
        elif not scale_matches:
            row['reason'] = 'Unresolved unit or scale conflict; no silent rescaling'
        else:
            lo, hi = Decimal(r['low']), Decimal(r['high']); assert lo <= hi
            emin, emax = max(lo-pred, Decimal(0), pred-hi), max(abs(pred-lo), abs(pred-hi))
            row.update(minimum_error_units=str(emin), maximum_error_units=str(emax),
                       minimum_error_fullscale=str(emin/span), maximum_error_fullscale=str(emax/span))
            if emax <= tol:
                row.update(classification='definitely_within_tolerance', reason='Entire possible error interval is within tolerance')
            elif emin > tol:
                row.update(classification='definitely_outside_tolerance', reason='Entire possible error interval exceeds tolerance')
            else:
                row['reason'] = 'Reference permits both within-tolerance and outside-tolerance readings'
        rows.append(row)
    assert len(rows) == 9 and len({r['file'] for r in rows}) == 9
    counts = Counter(r['classification'] for r in rows)
    result = {'status':'human_review_complete_comparison_complete', 'reviewer':reference['reviewer'],
              'human_review_exposure_note':reference['exposure_note'],
              'scope':'Assisted reading with manually supplied per-image geometry; single human reviewer; no physical instrument truth.',
              'accepted_cases':len(rows), 'definitely_within_tolerance':counts['definitely_within_tolerance'],
              'definitely_outside_tolerance':counts['definitely_outside_tolerance'], 'unresolved':counts['unresolved'],
              'original_abstentions':sum(p.get('prediction',{}).get('decision') == 'abstain' for p in predictions.values()),
              'original_source_failures':sum(p['execution'] == 'source_unavailable' for p in predictions.values()),
              'incorrect_acceptance_rate_identifiable':counts['unresolved'] == 0,
              'decision':'Stop accuracy evaluation on this dataset' if counts['unresolved'] == len(rows) else 'Report only decidable cases; no deployment reliability claim',
              'reader_runs':0, 'retuning':False, 'reference_intervals_narrowed_by_agent':False,
              'unit_amendment':'R02 MP to MPa explicitly confirmed by user; no prediction/interval change',
              'inputs_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [packet/'final_reference_record.json',packet/'human_finalization_confirmation.json',old/'predictions.json',old/'frozen_reader.py']}}
    for name, data in [('human_classification.json',result),('human_case_classifications.json',rows)]:
        with (packet/name).open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    with (packet/'human_case_classifications.csv').open('x') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('packet',type=Path)
    main(p.parse_args().packet)
