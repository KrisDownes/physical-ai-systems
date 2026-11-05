"""Independent truth-only evaluation of fixed recorded-scan radius comparisons."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from continuous_asset_analysis import truth_pose

OUT=ROOT/'results/radius_limit_20261026'
SOURCES={
    'fresh_moved_wrong':ROOT/'results/constraint_repair_20261026/fresh/mission_fresh_moved_wrong',
    'occluded':ROOT/'results/connected_layout_20261025/mission_occluded',
    'regression_visible':ROOT/'results/connected_layout_20261025/mission_visible',
}

def at(t,truth):return np.array([np.interp(t,truth[:,0],truth[:,i]) for i in (1,2,3)])
def main():
    result=[]
    for case,mission in SOURCES.items():
        truth=truth_pose(mission)
        for tag,radius in [('3m',3.0),('1p5m',1.5)]:
            stem=f'{case}_{tag}'
            if case=='fresh_moved_wrong':stem+='_v2'
            rows=[q.strip().split(',') for q in (OUT/f'{stem}.log').read_text().splitlines() if q.startswith('POSE,')]
            poses={int(q[1]):np.array(list(map(float,q[2:]))) for q in rows}
            assert list(poses)==list(range(len(poses)))
            err=[float(np.linalg.norm(p[1:3]-2-at(p[0],truth)[:2])) for p in poses.values()]
            with (OUT/f'{stem}_events.tsv').open() as f:events=list(csv.DictReader(f,delimiter='\t'))
            added=[e for e in events if e['event']=='EDGE_ADDED' and int(e['other'])<int(e['scan'])-10]
            true_links=[];wrong_links=[]
            for e in added:
                u,v=int(e['other']),int(e['scan']);a,b=at(poses[u][0],truth),at(poses[v][0],truth)
                d=b[:2]-a[:2];c,s=math.cos(a[2]),math.sin(a[2]);actual=(c*d[0]+s*d[1],-s*d[0]+c*d[1])
                error=math.dist(actual,(float(e['x']),float(e['y'])))
                item=dict(source=u,target=v,relative_translation_error_m=round(error,4),physical_separation_m=round(float(np.linalg.norm(d)),4))
                (wrong_links if error>.5 else true_links).append(item)
            accepted=[e for e in events if e['event']=='ACCEPT_LOOP']
            checks=[e for e in events if e['event']=='MATCHER_CHECK' and e['detail'].startswith('COARSE RESPONSE:')]
            r=dict(case=case,radius_m=radius,processed_scans=len(poses),
                   endpoint_position_error_m=round(err[-1],4),max_position_error_m=round(max(err),4),
                   median_position_error_m=round(float(np.median(err)),4),
                   accepted_loop_callbacks=len(accepted),coarse_matches=len(checks),
                   true_nonlocal_links=true_links,wrong_nonlocal_links=wrong_links,
                   accepted_callbacks_without_new_nonlocal_link=len(accepted)-len([e for e in added if e['scan'] in {a['scan'] for a in accepted}]))
            if case=='fresh_moved_wrong':
                # Representative ordinary physical revisit fixed in protocol.
                u,v=96,153;a,b=poses[u],poses[v]
                cum=np.r_[0,np.cumsum(np.linalg.norm(np.diff(truth[:,1:3],axis=0),axis=1))]
                travel=float(np.interp(b[0],truth[:,0],cum)-np.interp(a[0],truth[:,0],cum))
                row=next(e for e in events if e['scan']==str(v) and e['other']==str(u) and e['radius']==str(int(radius) if radius==3 else radius) and e['event'] in ['LINKED_EXCLUSION','DISTANCE_ADMITTED','DISTANCE_EXCLUDED'])
                r['physical_revisit']=dict(source=u,target=v,time_gap_s=round(float(b[0]-a[0]),3),
                    intervening_path_m=round(travel,3),physical_separation_m=round(float(np.linalg.norm(at(a[0],truth)[:2]-at(b[0],truth)[:2])),3),
                    estimated_reference_separation_m=round(float(row['separation_m']),3),candidate_status=row['event'])
                r['accepted_post_match_nearest']=[dict(scan=int(e['scan']),prior=int(e['other']),distance_m=round(float(e['separation_m']),4)) for e in events if e['event']=='POST_MATCH_NEAREST']
            result.append(r)
    live=json.loads((SOURCES['fresh_moved_wrong']/'outcome.json').read_text())
    primary={r['radius_m']:r for r in result if r['case']=='fresh_moved_wrong'}
    three_only={(e['source'],e['target']) for e in primary[3.0]['true_nonlocal_links']} - \
               {(e['source'],e['target']) for e in primary[1.5]['true_nonlocal_links']}
    narrow_worse=primary[1.5]['endpoint_position_error_m']-primary[3.0]['endpoint_position_error_m']>=.3
    trigger=bool(three_only) and narrow_worse
    summary=dict(protocol='docs/radius_limit/protocol.md',recorded_input_results=result,
                 prior_frozen_1p5m_live_mission=dict(complete_contract=live['complete_contract'],false_completion=live['false_completion'],
                     wrong_asset_acceptances=live['wrong_asset_acceptances'],handoff_seed_position_error_m=live['handoff_seed_position_error_m'],
                     total_sim_s=live['total_sim_s'],total_path_m=live['total_path_m'],contacts=live['contacts'],clearance_m=live['clearance_m']),
                 decision_gate=dict(correct_3m_only_inserted_links=sorted(map(list,three_only)),narrow_endpoint_worse_by_0_3m=narrow_worse,
                                    conditional_full_mission_pair_triggered=trigger),
                 limitation='Offline mapper has no Ceres optimizer and uses scans selected by the original 1.5m live acquisition; 3m closed-loop outcome remains untested')
    (OUT/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    for r in result:print(r['case'],r['radius_m'],'endpoint',r['endpoint_position_error_m'],'accepted',r['accepted_loop_callbacks'],'inserted true/wrong',len(r['true_nonlocal_links']),len(r['wrong_nonlocal_links']))

if __name__=='__main__':main()
