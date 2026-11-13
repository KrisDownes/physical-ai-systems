"""Vision-based nominal pick/place; privileged scoring never drives actions."""
import time
import numpy as np
from environment import cartesian_step, locate_green, pixel_to_plane, recalibrate_translation, OnlineJacobian, GlobalCorrection


def run(env, object_xy, q=None, method='nominal', seed=0):
    obs = env.reset(object_xy, q, seed=seed)
    start = time.perf_counter()
    pixel = locate_green(obs['image'])
    env.compute_s += time.perf_counter()-start
    result = {'visible': pixel is not None, 'stages': {}, 'frames': [obs['image']],
              'tracking_error_m_s': 0., 'peak_tracking_error_m': 0.}
    if pixel is None:
        result.update(success=False, failure='perception')
        return result
    begin = time.perf_counter()
    offset = recalibrate_translation(obs['image']) if method == 'recalibration' else np.zeros(2)
    env.compute_s += time.perf_counter()-begin
    if offset is None:
        result.update(success=False, failure='calibration')
        return result
    obj = pixel_to_plane(pixel, offset_xy=offset)
    online = OnlineJacobian() if method == 'online' else None
    global_model = GlobalCorrection(gp=(method=='gp')) if method in ('global','gp') else None
    grasp_xy = obj[:2] + np.array([-.01161,-.01240])

    def move(name, target, grip, budget=70):
        nonlocal obs
        for _ in range(budget):
            start = time.perf_counter()
            pos,_ = env.controller_fk(obs['joint_rad'])
            err=float(np.linalg.norm(target-pos))
            env.compute_s += time.perf_counter()-start
            result['tracking_error_m_s'] += err*.1
            result['peak_tracking_error_m'] = max(result['peak_tracking_error_m'],err)
            if err < .003:
                result['stages'][name] = True
                return True
            obs = cartesian_step(env,obs,target-pos,grip,online,global_model)
            if env.action_count % 5 == 0: result['frames'].append(obs['image'])
        result['stages'][name] = False
        result['last_error_m'] = float(np.linalg.norm(target-pos))
        return False

    if not move('approach', np.r_[grasp_xy,.06], 1.2) or not move('reach', np.r_[grasp_xy,.014], 1.2):
        result.update(success=False, failure='reach')
        return result
    for _ in range(15): obs = env.step(np.zeros(5), -.1)
    env.mark_evaluation('after_close')
    if not move('lift', np.r_[grasp_xy,.06], -.1):
        result.update(success=False, failure='lift_command', evaluation=env.evaluate())
        return result
    env.mark_evaluation('after_lift')
    destination = np.array([.22-.01161+.01,-.05-.0124+.02,.06])
    if not move('transport', destination, -.1) or not move('lower', destination + [0,0,-.046], -.1):
        result.update(success=False, failure='transport', evaluation=env.evaluate())
        return result
    for _ in range(35): obs=env.step(np.zeros(5),1.2)
    env.mark_evaluation('after_settle')
    score = env.evaluate()
    result['frames'].append(env.observe()['image'])
    failure = 'grasp' if not score['grasp_contact'] else 'lift' if not score['lifted'] else 'placement' if not score['placed'] else None
    result.update(success=score['success'], failure=failure, evaluation=score)
    return result
