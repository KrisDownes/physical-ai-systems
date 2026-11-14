"""Frozen fresh paired evaluation of camera calibration and visual self-calibration."""
import json
import argparse
import hashlib
import time
import uuid
from pathlib import Path
import numpy as np
from environment import Environment,Q_HOME
from nominal import run
from run_store import create_run

PLAN=Path('so101/frozen_camera_followup.json')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',help='Resume a partial run after verifying its frozen configuration and controller source')
    args=parser.parse_args()
    plan=json.loads(PLAN.read_text())
    if args.resume:
        out=Path(args.resume)
        meta=json.loads((out/'metadata.json').read_text())
        if meta['config']!=plan: raise ValueError('frozen configuration differs from partial run')
        for name in ('so101/environment.py','so101/nominal.py','so101/frozen_camera_followup.json'):
            digest=hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if meta['source_sha256'][name]!=digest: raise ValueError(f'controller source changed: {name}')
        if (out/'summary.json').exists(): raise FileExistsError('run already completed')
        stamp={'resumed_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               'resume_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        with (out/f'resume_{uuid.uuid4().hex[:8]}.json').open('x') as f:json.dump(stamp,f,indent=2)
    else:
        out=create_run('fresh_camera_followup',plan)
        (out/'episodes').mkdir()
    rows=[json.loads(p.read_text()) for p in sorted((out/'episodes').glob('*.json'))]
    completed={(r['case']['id'],r['condition'],r['method']) for r in rows}
    schedule=[(case,mode,case['camera_shift_m']) for case in plan['cases'] for mode in plan['reference_modes']]
    schedule += [(case,'visible',[0.,0.]) for case in plan['cases'] if case['id'] in plan['no_change_cases']]
    for case,mode,shift in schedule:
        condition='no_change' if shift==[0.,0.] else mode
        for method in plan['methods']:
            if (case['id'],condition,method) in completed: continue
            env=Environment(camera_shift=shift,reference_mode=mode,tool_marker=True,
                            action_budget=plan['max_actions'],movement_budget=plan['movement_budget_rad'])
            q=Q_HOME.copy();q[0]=case['initial_pan'];started=time.perf_counter()
            try:
                result=run(env,case['object_xy'],q,method=method,seed=case['seed'])
            except RuntimeError as error:
                if 'budget exhausted' not in str(error): raise
                result={'success':False,'failure':'budget','error':str(error)}
            result.pop('frames',None)
            visibility=result.get('initial_visibility',{})
            eligible=bool(visibility.get('object') and visibility.get('gripper_marker'))
            record={'case':case,'condition':condition,'reference_mode':mode,'camera_shift_m':shift,
                    'method':method,'result':result,'initial_sensor_eligible':eligible,
                    'actions':env.action_count,'movement_rad':env.movement,
                    'simulation_time_s':float(env.data.time),'wall_s':time.perf_counter()-started,
                    'controller_compute_s':env.compute_s,'limit_violations':env.limit_violations,
                    'saturated_actions':env.saturated_actions,'retries':0,
                    'self_collision_physics_steps':env.physics_collision_steps,
                    'self_collision_body_pairs':env.physics_collision_pairs,'trace':env.trace}
            rows.append(record)
            filename=f"{condition}_{case['id']}_{method}.json"
            with (out/'episodes'/filename).open('x') as f:json.dump(record,f,indent=2)
            print(condition,case['id'],method,result['success'],result.get('failure'),'eligible',eligible,flush=True)
            env.close()
    summary={}
    for condition in ['no_change']+plan['reference_modes']:
        summary[condition]={}
        for method in plan['methods']:
            group=[r for r in rows if r['condition']==condition and r['method']==method]
            failures={}
            for r in group:
                if not r['result']['success']:
                    reason=r['result'].get('failure','unknown');failures[reason]=failures.get(reason,0)+1
            summary[condition][method]={'trials':len(group),'successes':sum(r['result']['success'] for r in group),
                'ineligible_visibility':sum(not r['initial_sensor_eligible'] for r in group),'failures':failures,
                'median_actions':float(np.median([r['actions'] for r in group])),
                'median_movement_rad':float(np.median([r['movement_rad'] for r in group])),
                'median_simulation_s':float(np.median([r['simulation_time_s'] for r in group])),
                'median_wall_s':float(np.median([r['wall_s'] for r in group])),
                'self_collision_physics_steps':sum(r['self_collision_physics_steps'] for r in group)}
    with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2));print('saved:',out)


if __name__=='__main__':main()
