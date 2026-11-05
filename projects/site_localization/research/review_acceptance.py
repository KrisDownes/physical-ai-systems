"""Reproduce the review's existing-screen audit and figure, without ROS."""
import argparse
import json
from pathlib import Path
import numpy as np
from frames import compose, inverse
from register_maps import transform
from transfer_acceptance import analyze_alignment
from transfer_report import mission

ROOT = Path(__file__).resolve().parents[1]


def run(out):
    plan = json.loads((ROOT/'results/correction_surface_policy_20261022/plan.json').read_text())
    rows, paired = [], []
    for job in plan['jobs']:
        p = Path(job['acquisition'])
        a = json.loads((p/'surface_alignment.json').read_text())
        result = analyze_alignment(a, p/'source_surface.txt', p/'target_surface.txt')
        d = result['diagnostics']
        cfg = json.loads((p.parent/p.name.replace('__acquisition', '__deployment')/'controller_config.json').read_text())
        goals = np.array(cfg['goals'])
        map_goals = compose(inverse(a['target_from_source']), goals)[:, :2]
        spread = max(np.linalg.norm(transform(map_goals, np.array(a['candidates'][c['candidate_index']]['pose']))
                                   - goals[:, :2], axis=1).max() for c in d['near_optimal_candidates'])
        rows.append(dict(layout=job['layout'], repeat=job['repeat'], rules=result['rules'],
            retained_bin_fraction=d['retained_target_bin_fraction'],
            max_sampled_near_tie_task_displacement_m=float(spread),
            max_sampled_near_tie_map_corner_displacement_m=max(c['max_map_corner_displacement_m'] for c in d['near_optimal_candidates']),
            rank=a['conditioning']['rank'], condition=a['conditioning']['condition_number']))
        for policy, path in [('dock', p.parent/p.name.replace('__acquisition', '__deployment')),
                             ('surface', Path(job['artifact_directory']))]:
            r = mission(path)
            paired.append(dict(layout=job['layout'], repeat=job['repeat'], policy=policy,
                artifact_directory=str(path), **{k: r[k] for k in [
                    'mission_success', 'physical_stops', 'dock_after_completed_inspections',
                    'physical_dock_after_last_claim', 'physical_dock_dwell_s',
                    'max_claim_position_m', 'max_claim_heading_deg']}))
    (out/'acceptance_audit.json').write_text(json.dumps(rows, indent=2)+'\n')
    (out/'paired_mission_audit.json').write_text(json.dumps(paired, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rankings = json.loads((out/'sensor_rankings.json').read_text())
    evaluations = json.loads((out/'diagnostic_evaluation.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    r = rankings[1]
    old, new = r['candidates'][0], r['candidates'][r['winners']['capped_m2']]
    for offset, row, label, color in [(-.17, old, 'Trimmed winner', '#b44a3a'), (.17, new, 'All-source winner', '#287a9c')]:
        axes[0].bar(np.arange(2)+offset,
                    [100*row[k]['rms_m'] for k in ['baseline_retained','baseline_discarded']],
                    width=.34, label=label, color=color)
    axes[0].set_xticks([0, 1], ['Selected 70%\n(same cells)', 'Discarded 30%\n(same cells)'])
    axes[0].set_ylabel('Surface residual RMS (cm)')
    axes[0].set_title('Maintenance repeat 1: evidence lost by trimming')
    axes[0].legend(fontsize=9)
    for offset, key, label, color in [(-.17,'trimmed_m2','Trimmed winner','#b44a3a'),(.17,'capped_m2','All-source winner','#287a9c')]:
        axes[1].bar(np.arange(4)+offset,
            [100*e['calibration_only_on_fixed_hard_dock_trace'][key]['all_claims']['max_m'] for e in evaluations],
            width=.34, color=color, label=label)
    axes[1].set_xticks(range(4), ['Maint. 0','Maint. 1','Workshop 0','Workshop 1'])
    axes[1].set_ylabel('Maximum calibration residual at six claims (cm)')
    axes[1].set_title('Fixed recorded traces; not new mission outcomes')
    fig.savefig(out/'information_ablation.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
