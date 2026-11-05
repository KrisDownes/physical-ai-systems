"""Narrow development map qualification: independent task outcomes, no promotion.

This compares completed simulation runs, not internal scan residuals. It is not
a safety certificate or a qualification of hardware/real deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(baseline,candidate):
    base=json.loads((baseline/'outcome.json').read_text())
    test=json.loads((candidate/'outcome.json').read_text())
    bc=json.loads((baseline/'cell.json').read_text());cc=json.loads((candidate/'cell.json').read_text())
    common=['start','goals','dwell_s','judge_position_m','judge_yaw_rad','max_sim_s','obstacles']
    if any(bc[k]!=cc[k] for k in common):raise ValueError('Task/world criteria differ; cannot compare these runs')
    if sha(baseline/'cell.sdf')!=sha(candidate/'cell.sdf'):raise ValueError('Physical worlds differ; requires a separate scenario protocol')
    bh=json.loads((baseline/'source_hashes.json').read_text());ch=json.loads((candidate/'source_hashes.json').read_text())
    for name in ['sim/mission_node.py','sim/judge_node.py','sim/score_mission.py','vendor/offline_amcl','vendor/offline_amcl.cpp']:
        if bh[name]!=ch[name]:raise ValueError(f'Changed consequential component: {name}')
    if not base['mission_success']:raise ValueError('Reference mission is not a functioning baseline')
    reasons=[]
    if not test['mission_success']:reasons.append('Candidate fails independent physical mission criteria')
    if test['physical_stops_completed']<base['physical_stops_completed']:reasons.append('Fewer required physical stops completed')
    if test['body_contact_messages']:reasons.append('Body contacts observed')
    if not test['contact_system_live']:reasons.append('Contact instrumentation not live')
    result=dict(scope='one development scenario; no physical or safety qualification',
                decision='reject' if reasons else 'eligible_for_further_testing',reasons=reasons,
                baseline_stops=base['physical_stops_completed'],candidate_stops=test['physical_stops_completed'],
                baseline_map_sha256=sha(baseline/'map.txt'),candidate_map_sha256=sha(candidate/'map.txt'),
                physical_world_sha256=sha(baseline/'cell.sdf'),map_promoted=False,
                baseline_outcome_sha256=sha(baseline/'outcome.json'),candidate_outcome_sha256=sha(candidate/'outcome.json'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=qualify(a.baseline,a.candidate);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
