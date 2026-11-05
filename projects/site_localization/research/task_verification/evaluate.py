"""Join previously recorded image-only predictions to render ground truth."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main(root):
    predictions = json.loads((root/'decisions.json').read_text())
    truth = {r['id']: r for r in json.loads((root/'evaluation_truth.json').read_text())}
    rows, summary = [], {}
    for method in ['arrival', 'reference_reading', 'tag_only', 'tag_reading']:
        ms = {}
        for group in ['all', 'trimmed', 'capped']:
            counts = dict(n=0, positive=0, negative=0, accepted=0, false_accept=0,
                          false_reject_including_abstention=0, abstained=0,
                          accepted_readings=0, inaccurate_accepted_readings=0)
            errs = []
            for p in predictions:
                gt, pred = truth[p['id']], p[method]
                if group != 'all' and gt['source_group'] != group:
                    continue
                accept, positive = pred['decision'] == 'accept', gt['task_positive']
                err = abs(pred['reading_fraction']-gt['fraction']) if 'reading_fraction' in pred else None
                counts['n'] += 1
                counts['positive' if positive else 'negative'] += 1
                counts['accepted'] += int(accept)
                counts['false_accept'] += int(accept and not positive)
                counts['false_reject_including_abstention'] += int(not accept and positive)
                counts['abstained'] += int(pred['decision'] == 'abstain')
                if accept and err is not None:
                    counts['accepted_readings'] += 1
                    counts['inaccurate_accepted_readings'] += int(err > .02)
                    errs.append(err)
                if group == 'all':
                    rows.append(dict(id=p['id'], group=gt['source_group'], condition=gt['condition'],
                                     method=method, positive=positive, decision=pred['decision'],
                                     reason=pred['reason'], reading_fraction=pred.get('reading_fraction'),
                                     absolute_fraction_error=err))
            counts.update(coverage=counts['accepted']/counts['n'],
                          max_accepted_fraction_error=max(errs) if errs else None)
            ms[group] = counts
        summary[method] = ms
    result = {'scope': 'Analytic camera retrofit: 10 images, 2 source trajectories, 1 panel fixture; no new missions/physical trials',
              'methods': summary,
              'timing': {key: sum(p['timing'][key] for p in predictions) for key in predictions[0]['timing']},
              'uncertainty': 'Finite controlled examples; no frame/binomial reliability interval',
              'decision_sha256': hashlib.sha256((root/'decisions.json').read_bytes()).hexdigest()}
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    with (root/'outcomes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    main(p.parse_args().root)
