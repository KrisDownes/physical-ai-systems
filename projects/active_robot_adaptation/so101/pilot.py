"""Bounded nominal development pilot; no recovery comparison until readiness."""
import json
import platform
import time
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from environment import Environment, Q_HOME
from nominal import run
from run_store import create_run

# Frozen before this sweep; the first case was used while debugging.
CASES = [(.22,.024,-.60), (.215,.015,-.55), (.225,.035,-.65),
         (.21,.035,-.55), (.23,.015,-.65), (.225,.02,-.60)]


def main():
    OUT = create_run('nominal', {'cases': CASES, 'seeds': list(range(len(CASES)))})
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    env = Environment()
    for i,(x,y,pan) in enumerate(CASES):
        q=Q_HOME.copy(); q[0]=pan
        started=time.perf_counter()
        result=run(env,(x,y),q,seed=i)
        elapsed=time.perf_counter()-started
        frames=result.pop('frames')
        if i == 0 or not result['success']:
            imageio.mimsave(OUT/f'episode_{i}.gif',frames,duration=200,loop=0)
        result.update(case=i, initial_object_xy=[x,y], initial_pan=pan, seed=i,
                      action_count=env.action_count, movement_rad=env.movement,
                      limit_violations=env.limit_violations, saturated_actions=env.saturated_actions,
                      simulation_s=float(env.data.time), controller_compute_s=env.compute_s,
                      wall_s=elapsed)
        rows.append(result)
        print('case',i,'success',result['success'],'failure',result['failure'],'wall',round(elapsed,2),flush=True)
    env.close()
    result={'cases':rows,'readiness':{'criterion':'at least 5 successes in 6 development episodes',
            'passed':sum(r['success'] for r in rows)>=5},
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'mujoco':mujoco.__version__},
            'model_revision':'370c43965323c0944b433b202aac5058b5f907a9'}
    (OUT/'nominal_development.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['readiness']))


if __name__ == '__main__': main()
