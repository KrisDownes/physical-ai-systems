"""Run the frozen paired comparison with full independent resets."""
import json
import argparse
import platform
import subprocess
import time
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from environment import Environment, Q_HOME
from nominal import run
from run_store import create_run

BASELINE = Path('results/so101')
PLAN = Path('so101/frozen_comparison.json')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--transfer',action='store_true')
    args=parser.parse_args()
    nominal = json.loads((BASELINE/'nominal_development.json').read_text())
    if not nominal['readiness']['passed']:
        raise RuntimeError('Nominal readiness gate not met; no disturbance comparison')
    plan=json.loads(PLAN.read_text())
    label='comparison'
    if args.transfer:
        transfer=json.loads(Path('so101/frozen_transfer.json').read_text())
        plan['methods']=transfer['methods']
        plan['transfer']=transfer
        label='transfer'
    OUT=create_run(label,plan)
    rows=[]
    traces={}
    OUT.mkdir(parents=True,exist_ok=True)
    for condition in plan['conditions']:
        env=Environment(camera_shift=condition['camera_shift_m'],joint_gain=condition['joint_gain'])
        for case in plan['cases']:
            for method in plan['methods']:
                q=Q_HOME.copy();q[0]=case['initial_pan']
                start=time.perf_counter()
                r=run(env,case['object_xy'],q,method=method,seed=case['seed'])
                elapsed=time.perf_counter()-start
                frames=r.pop('frames')
                if case['id']==0 and method=='nominal':
                    imageio.mimsave(OUT/f"{condition['name']}_nominal.gif",frames,duration=200,loop=0)
                r.update(case=case,condition=condition['name'],method=method,actions=env.action_count,
                         movement_rad=env.movement,simulation_s=float(env.data.time),
                         recovery_time_sim_s=float(env.data.time) if r['success'] else None,
                         retries=0,limit_violations=env.limit_violations,saturated_actions=env.saturated_actions,
                         controller_compute_s=env.compute_s,wall_s=elapsed)
                traces[f"{condition['name']}-{case['id']}-{method}"]=env.trace
                rows.append(r)
                print(condition['name'],case['id'],method,r['success'],r['failure'],flush=True)
        env.close()
    summary={}
    for cond in plan['conditions']:
        summary[cond['name']]={}
        for method in plan['methods']:
            group=[r for r in rows if r['condition']==cond['name'] and r['method']==method]
            summary[cond['name']][method]={'successes':sum(r['success'] for r in group),'trials':len(group),
                'failures':{name:sum(r['failure']==name for r in group) for name in
                            ['perception','calibration','reach','grasp','lift','lift_command','transport','placement']},
                'median_actions':float(np.median([r['actions'] for r in group])),
                'median_movement_rad':float(np.median([r['movement_rad'] for r in group])),
                'median_wall_s':float(np.median([r['wall_s'] for r in group]))}
    output={'plan':plan,'episodes':rows,'summary':summary,
            'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'mujoco':mujoco.__version__}}
    (OUT/f'{label}.json').write_text(json.dumps(output,indent=2)+'\n')
    (OUT/f'{label}_traces.json').write_text(json.dumps(traces,separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2))
    print('saved:',OUT)


if __name__=='__main__': main()
