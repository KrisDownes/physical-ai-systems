"""Recover physics-step timing by replaying existing commands and test one actuator change."""

import argparse
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

from compare import wrap, yaw
from iteration import source
from review_velocity import PUSH, THETA
from run_store import create


def runs(mask, dt):
    """Half-open time intervals relative to the first push command."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(float(i*dt), float(j*dt)) for i, j in edges.reshape(-1, 2)]


def replay(make_env, row, interior_target_mm=None):
    robot = yaml.safe_load(Path('configs/robot/sim_2_dof.yaml').read_text())
    obj = yaml.safe_load(Path('configs/env/rod.yaml').read_text())
    params = yaml.safe_load(Path('configs/asid/inertia.yaml').read_text())
    params['reward'] = False
    env = make_env(robot, obj, params, seed=row['seed'])
    m = env.unwrapped._robot.model
    m.geom_rgba[m.geom('rod_visual_com').id, 3] = 0
    env.parameter_dict['inertia']['value'] = row[THETA]
    env.params_set = True
    np.random.seed(row['seed'])
    observations = [env.reset().copy()]
    poses = [env.get_obj_pose().copy()]
    names = ('robot:finger_joint1', 'robot:finger_joint2')
    joints = [m.joint(name).id for name in names]
    addresses = m.jnt_qposadr[joints]
    lo, hi = m.jnt_range[joints].T
    finger_bodies = set(m.jnt_bodyid[joints])
    robot_bodies = {m.body(name).id for name in (
        *(f'link{i}' for i in range(8)), 'hand', 'left_finger', 'right_finger')}
    rod_geom = m.geom('rod_geom_0').id
    actuator = m.actuator('panda_actuator8').id
    dt = m.opt.timestep
    step_data = []
    original_step = mujoco.mj_step

    def observe_step(model, data, nstep=1):
        for _ in range(nstep):
            if interior_target_mm is not None:
                # The actuator tendon averages the two finger positions.
                # Map an interior position to input using the model's gain/bias.
                data.ctrl[actuator] = (-m.actuator_biasprm[actuator, 1]
                                       * interior_target_mm / 1000
                                       / m.actuator_gainprm[actuator, 0])
            original_step(model, data)
            qpos = data.qpos[addresses]
            robot_contact = finger_contact = False
            for contact in data.contact[:data.ncon]:
                if contact.dist > 0 or rod_geom not in (contact.geom1, contact.geom2):
                    continue
                other = contact.geom2 if contact.geom1 == rod_geom else contact.geom1
                body = int(m.geom_bodyid[other])
                robot_contact |= body in robot_bodies
                finger_contact |= body in finger_bodies
            step_data.append([*np.maximum(lo - qpos, 0), *np.maximum(qpos - hi, 0),
                              robot_contact, finger_contact])

    try:
        mujoco.mj_step = observe_step
        for action in row['actions']:
            observations.append(env.step(np.array(action))[0].copy())
            poses.append(env.get_obj_pose().copy())
    finally:
        mujoco.mj_step = original_step

    steps = np.array(step_data)
    contact = steps[:, 4].astype(bool)
    finger_contact = steps[:, 5].astype(bool)
    limit = np.max(steps[:, :4], axis=1) > 1e-5
    first, last = np.flatnonzero(contact)[[0, -1]] if contact.any() else (len(limit), -1)
    idx = np.arange(len(limit))
    maximum = np.max(steps[:, :4], axis=1)
    max_during = maximum[finger_contact].max(initial=0)
    durations = {
        'before_first_contact_steps': int(sum(limit & (idx < first))),
        'in_robot_contact_steps': int(sum(limit & contact)),
        'between_contacts_steps': int(sum(limit & ~contact & (idx >= first) & (idx <= last))),
        'after_last_contact_steps': int(sum(limit & (idx > last) & (idx >= first))),
        'in_finger_contact_steps': int(sum(limit & finger_contact)),
    }
    if interior_target_mm is None:
        np.testing.assert_array_equal(observations, row['observations'])
        np.testing.assert_array_equal(poses, row['rod_pose_evaluator'])
        assert int(sum(limit)) == row['audit']['limit_steps']
        assert int(sum(contact)) == row['audit']['rod_robot_contact_steps']
        assert len(limit) == row['audit']['steps']
    return {
        'case': row['case'], 'interior_target_mm': interior_target_mm, 'dt_s': dt,
        'steps': len(limit), 'limit_steps': int(sum(limit)), 'contact_steps': int(sum(contact)),
        'max_lower_mm': (1000 * steps[:, :2].max(axis=0)).tolist(),
        'max_upper_mm': (1000 * steps[:, 2:4].max(axis=0)).tolist(),
        'max_during_finger_contact_mm': float(1000*max_during),
        'limit_intervals_s': runs(limit, dt), 'contact_intervals_s': runs(contact, dt),
        'finger_contact_intervals_s': runs(finger_contact, dt), **durations,
        'yaw_response_rad': float(wrap(yaw(poses[-1][3:7]) - yaw(poses[0][3:7]))),
        'final_rod_center_m': poses[-1][:3].tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=PUSH)
    args = p.parse_args()
    data_path = args.data.resolve()
    rows = json.loads(data_path.read_text())['episodes']
    # Include case 6 and the five cases with the largest recorded excursion.
    selected = {6} | {r['case'] for r in sorted(
        rows, key=lambda r: r['audit']['max_limit_excursion'], reverse=True)[:5]}
    out = create('rod_finger_review', {'data': str(data_path),
                 'scope': 'instrumentation replay of saved commands and selected actuation counterfactuals',
                 'interior_target_mm': 1., 'counterfactual_cases': sorted(selected)})
    source(out)
    upstream = Path(os.environ.get('ASID_SRC', '/tmp/active-asid-src')).resolve()
    sys.path.insert(0, str(upstream))
    from env.wrappers.asid_vec import make_env
    os.chdir(upstream)
    results = []
    for row in rows:
        results.append(replay(make_env, row))
        if row['case'] in selected:
            results.append(replay(make_env, row, interior_target_mm=1.))
        if row['case'] % 8 == 0:
            print('replayed case', row['case'], flush=True)
    (out / 'summary.json').write_text(json.dumps(results, separators=(',', ':')) + '\n')
    print(out, flush=True)


if __name__ == '__main__':
    main()
