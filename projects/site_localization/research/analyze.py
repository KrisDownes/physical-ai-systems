"""Offline development comparison. Simulator truth is evaluation-only."""
import argparse
import csv
import json
from pathlib import Path


def summarize(path):
    outcome=json.loads((path/'outcome.json').read_text())
    run=json.loads((path/'run.json').read_text())
    cfg=json.loads((path/'cell.json').read_text())
    errors={e['goal']:e for e in outcome['claimed_stop_errors'] if e['reference_supported']}
    pairs=[]
    if cfg.get('return_loop'):
        for before,after in [(0,5),(1,4)]:
            if before in errors and after in errors:
                pairs.append(dict(before=before,after=after,
                    position_change_m=errors[after]['physical_position_error_m']-errors[before]['physical_position_error_m']))
    return dict(run=path.name,condition=cfg['condition'],localizer=cfg['localizer'],mapping=cfg['mapping'],
        stage='candidate acquisition' if cfg['mapping'] in ['new','continued'] else 'fresh deployment',
        workflow_context=json.loads((path/'workflow_context.json').read_text()) if (path/'workflow_context.json').exists() else None,
        alignment_preparation=json.loads((path/'alignment_provenance.json').read_text()) if (path/'alignment_provenance.json').exists() else None,
        controller_claim_count=len(outcome['claimed_stop_errors']),
        physical_stops=outcome['physical_stops_completed'],required_stops=outcome['required_stops'],
        mission_success=outcome['mission_success'],dock_time_s=outcome.get('dock_completion_time_s'),
        max_claim_position_error_m=max((e['physical_position_error_m'] for e in errors.values()),default=None),
        max_claim_heading_error_deg=max((e['physical_heading_error_deg'] for e in errors.values()),default=None),
        remote_revisit_position_changes=pairs,claimed_stop_errors=outcome['claimed_stop_errors'],
        body_contacts=outcome['body_contact_messages'],clearance_m=outcome['minimum_conservative_clearance_m'],
        observed_sim_span_s=outcome['observed_sim_span_s'],wall_s=run['wall_s'],stop_reason=run['stop_reason'],
        children_cpu_s=run['children_cpu_user_s']+run['children_cpu_system_s'],
        largest_child_rss_kib=run['children_max_rss_kib'],serialization=run['serialization'],
        reset_count=run['reset_count'],operator_events=outcome['operator_events'],
        all_processes_reaped=run['all_owned_processes_reaped'],
        source_hashes=str(path/'source_hashes.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[summarize(path) for path in a.runs]
    (a.out/'comparison.json').write_text(json.dumps(dict(split='development',runs=rows,
        independence='Each row is one mission; all share a development layout and historical prior, and many share a commissioning graph. Revisited stops are paired, not independent trials.',
        uncertainty='No population-effect interval: one mission per method/condition. No held-out claim.',
        cost_limitations='Concurrent process wall times include contention; max RSS is largest child, not summed peak. SLAM runs include diagnostic AMCL. Human seconds are unmeasured.'),indent=2)+'\n')
    fields=['run','condition','localizer','mapping','physical_stops','required_stops','mission_success','dock_time_s',
        'max_claim_position_error_m','max_claim_heading_error_deg','body_contacts','clearance_m',
        'observed_sim_span_s','wall_s','children_cpu_s','largest_child_rss_kib','reset_count']
    with (a.out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5))
    for row in rows:
        errors=[e for e in row['claimed_stop_errors'] if e['reference_supported']]
        ax.plot([e['goal'] for e in errors],[e['physical_position_error_m'] for e in errors],marker='o',label=row['run'].replace('two_zone_loop_',''))
    ax.axhline(.15,color='black',linestyle='--',label='Position requirement (dwell/heading judged separately)')
    ax.set(xticks=range(6),xticklabels=['left0','left1','right0','right1','left1 return','left0 return'],
        ylabel='Physical position error at controller claim (m)',title='Development return-loop comparison; one mission per condition')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(a.out/'claim_errors.png',dpi=160)
    print(a.out/'comparison.json')


if __name__=='__main__':main()
