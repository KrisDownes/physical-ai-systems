"""Exploratory review of saved rods with fixed fits and the original release evaluator."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare import features, fit as fit_response, predict, wrap, yaw
from iteration import ROOT, outcome, source, training
from run_store import create
from velocity_check import feature, fit

RUNS = ROOT / 'results/published_runs'
PUSH = RUNS / '20261122T212216.364638Z_aa04a08d803e_rod_iteration_push_d192d75c/summary.json'
SAVED = RUNS / '20261122T212517.304304Z_aa04a08d803e_rod_velocity_posthoc_5f4741a5/summary.json'
PROTOCOL = Path(__file__).with_name('ITERATION_PROTOCOL.json')
THETA = 'hidden_mass_offset_m_evaluator_training'


def perfect_yaw(row):
    """Inject only ground-truth yaw, preserving observed tilt and other features."""
    observations = np.array(row['observations'])
    for obs, pose in zip(observations, row['rod_pose_evaluator']):
        q = obs[14:18].copy()
        delta = float(wrap(yaw(pose[3:7]) - yaw(q)))
        c, s = np.cos(delta / 2), np.sin(delta / 2)
        w, x, y, z = q
        obs[14:18] = [c*w - s*z, c*x - s*y, c*y + s*x, c*z + s*w]
    return {**row, 'observations': observations.tolist()}


def model_inputs(row, corrected, method):
    if method == 'D':
        original = features(row, 'history')
        exact = features(corrected, 'history')
        return original, {'exact_response': np.array([original[0], exact[1], original[2]]),
                          'exact_all_yaw': exact}
    original = feature(row, method)
    exact = feature(corrected, method)
    variants = {'exact_all_yaw': exact}
    if method == 'C':
        response = original.copy()
        # 25 current-state features + 2 initial geometry features, then 4 yaw changes.
        response[27:31] = exact[27:31]
        variants['exact_response'] = response
    return original, variants


def scored(row, model, inputs):
    choice = predict(model, inputs)
    return {'support_choice_m': choice, **outcome(row[THETA], choice)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--push', type=Path, default=PUSH)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    data = json.loads(args.push.read_text())['episodes']
    saved = json.loads(SAVED.read_text())['results']
    previous = {(row['case'], row['method']): row for row in saved}
    development = training()
    assert [(r['seed'], r[THETA]) for r in data] == [
        (c['seed'], c['theta']) for c in protocol['cases']]
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256(PROTOCOL.with_name(name).read_bytes()).hexdigest() == digest
    assert len(development) == 39 and len(data) == 48
    for key in ('seed', THETA):
        assert not ({r[key] for r in data} & {r[key] for r in development})
    models = {kind: fit(development, kind) for kind in 'BC'}
    models['D'] = fit_response(development, 'history')
    rows = []
    for row in data:
        observations = np.array(row['observations'])
        assert observations.shape == (21, 18) and np.shape(row['actions']) == (20, 2)
        b, c = feature(row, 'B'), feature(row, 'C')
        np.testing.assert_array_equal(np.r_[c[:25], c[-8:]], b)
        scrambled_truth = {**row, THETA: 9., 'rod_pose_evaluator': None, 'audit': None}
        for kind in 'BC':
            np.testing.assert_array_equal(feature(row, kind), feature(scrambled_truth, kind))
        corrected = perfect_yaw(row)
        observed_response = float(features(row, 'history')[1])
        exact_response = float(features(corrected, 'history')[1])
        result = {'case': row['case'], 'seed': row['seed'], 'com_m': .6*row[THETA],
                  'initial_observed_yaw_rad': float(yaw(observations[0, 14:18])),
                  'final_observed_yaw_rad': float(yaw(observations[-1, 14:18])),
                  'observed_response_rad': observed_response, 'exact_response_rad': exact_response,
                  'yaw_response_error_rad': float(wrap(observed_response - exact_response))}
        for kind, model in models.items():
            raw, variants = model_inputs(row, corrected, kind)
            base = scored(row, model, raw)
            if kind in 'BC':
                old = previous[row['case'], kind]
                assert abs(old['support_choice_m'] - base['support_choice_m']) < 1e-12
                assert old['stable_steps'] == base['stable_steps']
            result[kind] = {'observed': base, **{
                variant: scored(row, model, inputs) for variant, inputs in variants.items()}}
        rows.append(result)

    differences = 1000 * np.array([
        abs(r['C']['observed']['placement_error_evaluator_m'])
        - abs(r['B']['observed']['placement_error_evaluator_m']) for r in rows])
    rng = np.random.default_rng(443)
    sampled = differences[rng.integers(0, len(rows), (10000, len(rows)))]
    pairs = {label: [r['case'] for r in rows if (
        r['B']['observed']['success'], r['C']['observed']['success']) == state]
        for label, state in {'gained': (False, True), 'lost': (True, False),
                             'shared': (True, True), 'neither': (False, False)}.items()}
    oracle = {}
    for kind in 'BCD':
        for variant in ('exact_response', 'exact_all_yaw'):
            if variant not in rows[0][kind]:
                continue
            oracle[f'{kind}_{variant}'] = {
                'rescues': [r['case'] for r in rows if not r[kind]['observed']['success']
                            and r[kind][variant]['success']],
                'losses': [r['case'] for r in rows if r[kind]['observed']['success']
                           and not r[kind][variant]['success']],
            }
    summary = {
        'paired_mean_mm': float(differences.mean()),
        'paired_median_mm': float(np.median(differences)),
        'mean_ci95_mm': np.quantile(sampled.mean(axis=1), [.025, .975]).tolist(),
        'median_ci95_mm': np.quantile(np.median(sampled, axis=1), [.025, .975]).tolist(),
        'pairs': pairs, 'yaw_oracles': oracle, 'cases': rows,
    }
    out = create('rod_velocity_review', {'push': str(args.push.resolve()), 'original_correction': str(SAVED),
                 'scope': 'exploratory replay of saved release scores and yaw-only oracle diagnostics'})
    source(out)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'cases'}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
