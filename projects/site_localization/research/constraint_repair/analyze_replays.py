"""Evaluator-only comparison of Karto recorded-scan replays; truth never in replay."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from continuous_asset_analysis import truth_pose

def p_at(t,truth):return np.array([np.interp(t,truth[:,0],truth[:,i]) for i in (1,2,3)])
def main():
    output=ROOT/'results/constraint_repair_20261026';source=ROOT/'results/connected_layout_20261025'
    result=[]
    for case in ['visible','moved','occluded','wrong_asset','absent']:
        physical=truth_pose(source/f'mission_{case}')
        for policy in ['old','narrow']:
            stem=output/f'{case}_{policy}_replay'
            lines=stem.with_suffix('.log').read_text().splitlines()
            poses={int(a[1]):list(map(float,a[2:])) for line in lines if line.startswith('POSE,') for a in [line.split(',')]}
            assert sorted(poses)==list(range(len(poses)))
            errors=[]
            for i,(t,x,y,yaw) in poses.items():
                truth=p_at(t,physical);errors.append(float(np.linalg.norm(np.array([x-2,y-2])-truth[:2])))
            incorrect=[];legitimate=[]
            with (output/f'{case}_{policy}_replay_edges.csv').open() as f:
                for e in csv.DictReader(f):
                    u,v=int(e['source']),int(e['target'])
                    if v-u<10:continue
                    a=p_at(poses[u][0],physical);b=p_at(poses[v][0],physical)
                    delta=b[:2]-a[:2];c,s=math.cos(a[2]),math.sin(a[2]);actual=(c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1])
                    error=math.dist(actual,(float(e['dx']),float(e['dy'])))
                    record={'source':u,'target':v,'translation_error_m':round(error,4)}
                    (incorrect if error>.5 else legitimate).append(record)
            result.append(dict(case=case,policy=policy,radius_m=3.0 if policy=='old' else 1.5,
                               source='serialized processed scans and recorded odometry; no Gazebo; no solver',
                               endpoint_pose_error_m=round(errors[-1],4),max_pose_error_m=round(max(errors),4),
                               first_node_over_0_5m=next((i for i,e in enumerate(errors) if e>.5),None),
                               accepted_loop_events=sum(line.startswith('ACCEPT,') for line in lines),
                               incorrect_cross_time_links=incorrect,legitimate_cross_time_link_count=len(legitimate)))
    dest=output/'replay_comparison.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    for r in result:print(r['case'],r['policy'],r['endpoint_pose_error_m'],len(r['incorrect_cross_time_links']),r['legitimate_cross_time_link_count'],r['accepted_loop_events'])

if __name__=='__main__':main()
