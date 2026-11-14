"""Matched, independently reset integrated-controller experiments."""
import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
import numpy as np
from environment import Environment,Q_HOME
from nominal import run
from run_store import create_run
from diagnose import overlay


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume')
    args=parser.parse_args()
    cfgpath=Path(args.config); cfg=json.loads(cfgpath.read_text())
    if args.resume:
        out=Path(args.resume); meta=json.loads((out/'metadata.json').read_text())
        if meta['config']!=cfg:raise ValueError('frozen config differs')
        for name in ('so101/environment.py','so101/nominal.py','so101/calibration.py','so101/integrated_evaluation.py',str(cfgpath)):
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=meta['source_sha256'][name]:
                raise ValueError('source changed: '+name)
        if (out/'summary.json').exists():raise FileExistsError('run complete')
        with (out/f'resume_{uuid.uuid4().hex[:8]}.json').open('x') as f:
            json.dump({'time_utc':time.strftime('%FT%TZ',time.gmtime()),
                       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},f)
    else:
        out=create_run('integrated',cfg);(out/'episodes').mkdir()
    rows=[json.loads(p.read_text()) for p in sorted((out/'episodes').glob('*.json'))]
    done={(r['case']['id'],r['condition']['name'],r['method']) for r in rows}
    for cond in cfg['conditions']:
        for case in cfg['cases']:
            for method in cfg['methods']:
                if (case['id'],cond['name'],method) in done:continue
                shift=cond.get('camera_shift_m',case.get('camera_shift_m',[0,0]) if cond.get('camera_changed') else [0,0])
                tool=cond.get('tool_offset_wrist_m',case.get('tool_offset_wrist_m',[0,0,0]) if cond.get('tool_changed') else [0,0,0])
                actual_cond={**cond,'camera_shift_m':shift,'tool_offset_wrist_m':tool}
                env=Environment(camera_shift=shift,tool_offset=tool,
                  reference_mode=cond['references'],tool_marker=True,camera_noise_std=cfg['camera_noise_std'],
                  action_budget=cfg['action_budget'],movement_budget=cfg['movement_budget_rad'])
                q=Q_HOME.copy();q[0]=case['initial_pan'];start=time.perf_counter()
                record=(case==cfg['cases'][0] and method==cfg.get('record_method','integrated') and cond['name'] in cfg.get('record_conditions',[c['name'] for c in cfg['conditions']]))
                try:r=run(env,case['object_xy'],q,method=method,seed=case['seed'],diagnostics=record)
                except RuntimeError as error:
                    if 'budget exhausted' not in str(error):raise
                    r={'success':False,'failure':'budget','error':str(error)}
                frames=r.pop('diagnostic_frames',[]);r.pop('frames',None)
                if frames:overlay(frames,shift,out/f"{cond['name']}_{method}.gif")
                trace=env.trace
                accepted=sum(t.get('adaptation',{}).get('accepted',False) for t in trace)
                rejected=sum(not t['adaptation']['accepted'] for t in trace if 'adaptation' in t)
                snap=r.get('evaluation',{}).get('snapshots',{}).get('before_close',{})
                alignment=(np.asarray(snap['site_xyz'])-np.asarray(snap['object_xyz'])).tolist() if snap else None
                row={'case':case,'condition':actual_cond,'method':method,'result':r,'actions':env.action_count,
                   'movement_rad':env.movement,'simulation_s':float(env.data.time),'wall_s':time.perf_counter()-start,
                   'compute_s':env.compute_s,'accepted_motion_updates':accepted,'rejected_motion_updates':rejected,
                   'grasp_alignment_site_minus_object_m':alignment,'joint_limit_violations':env.limit_violations,
                   'joint_limit_physics_steps':env.physics_limit_steps,
                   'self_collision_physics_steps':env.physics_collision_steps,'self_collision_pairs':env.physics_collision_pairs,
                   'trace':trace}
                with (out/'episodes'/f"{cond['name']}_{case['id']}_{method}.json").open('x') as f:json.dump(row,f,indent=2)
                rows.append(row);env.close()
                print(cond['name'],case['id'],method,r['success'],r.get('failure'),flush=True)
    summary={}
    for cond in cfg['conditions']:
        summary[cond['name']]={}
        for method in cfg['methods']:
            group=[r for r in rows if r['condition']['name']==cond['name'] and r['method']==method]
            summary[cond['name']][method]={'successes':sum(r['result']['success'] for r in group),'trials':len(group),
              'failures':[r['result'].get('failure') for r in group if not r['result']['success']],
              'median_actions':float(np.median([r['actions'] for r in group])),
              'median_movement_rad':float(np.median([r['movement_rad'] for r in group])),
              'median_sim_s':float(np.median([r['simulation_s'] for r in group])),
              'median_compute_s':float(np.median([r['compute_s'] for r in group])),
              'median_tracking_error_m_s':float(np.median([r['result'].get('tracking_error_m_s',0.) for r in group])),
              'total_accepted_motion_updates':sum(r['accepted_motion_updates'] for r in group),
              'total_rejected_motion_updates':sum(r['rejected_motion_updates'] for r in group),
              'joint_limit_violations':sum(r['joint_limit_violations'] for r in group),
              'self_collision_physics_steps':sum(r['self_collision_physics_steps'] for r in group)}
    with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2));print('saved:',out)


if __name__=='__main__':main()
