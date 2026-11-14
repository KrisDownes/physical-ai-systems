"""Vision-based nominal pick/place; privileged scoring never drives actions."""
import time
import numpy as np
from environment import cartesian_step, locate_green, locate_cyan, pixel_to_plane, recalibrate_translation, OnlineJacobian, GlobalCorrection


def run(env, object_xy, q=None, method='nominal', seed=0, intervention=None, diagnostics=False):
    obs = env.reset(object_xy, q, seed=seed)
    start = time.perf_counter()
    pixel = locate_green(obs['image'])
    env.compute_s += time.perf_counter()-start
    result = {'visible': pixel is not None, 'stages': {}, 'frames': [obs['image']],
              'tracking_error_m_s': 0., 'peak_tracking_error_m': 0.}
    result['initial_visibility']={'object':pixel is not None,'gripper_marker':locate_cyan(obs['image']) is not None,
                                  'table_references':recalibrate_translation(obs['image']) is not None}
    if pixel is None:
        result.update(success=False, failure='perception')
        return result
    begin = time.perf_counter()
    integrated_family=method in ('fixed2','calibration','motion','integrated')
    online = OnlineJacobian() if method in ('online','motion','integrated') else None
    calibration_events=[]
    destination_pixel=None
    if integrated_family:
        from calibration import select_calibration,blue_destination,CalibrationState,VisualTargets
        destination_pixel=blue_destination(obs['image'])
        if destination_pixel is None:
            result.update(success=False,failure='destination_perception')
            return result
    offset = recalibrate_translation(obs['image']) if method == 'recalibration' else np.zeros(2)
    if method=='recalibration' and offset is None:
        # Up to 0.6 simulated seconds waiting for declared references; every step is charged.
        for _ in range(6):
            obs=env.step(np.zeros(5),obs['joint_rad'][5])
            if 'image' in obs:
                offset=recalibrate_translation(obs['image'])
                if offset is not None: break
    if method=='selfcal':
        marker=locate_cyan(obs['image'])
        if marker is None:
            offset=None
        else:
            env.controller_fk(obs['joint_rad'])
            import mujoco
            mid=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_SITE,'tool_marker_site')
            marker_xyz=env.nominal_data.site_xpos[mid].copy()
            offset=marker_xyz[:2]-pixel_to_plane(marker,z=marker_xyz[2])[:2]
    if method in ('calibration','integrated'):
        offset,calibration_events=select_calibration(env,obs)
        result['calibration_events']=calibration_events
        if offset is not None:
            state=CalibrationState();state.apply(offset,joint_model=online)
            calibration_events.extend(state.events)
    env.compute_s += time.perf_counter()-begin
    if offset is None:
        result.update(success=False, failure='calibration')
        return result
    obj = pixel_to_plane(pixel, offset_xy=offset)
    if integrated_family:
        obj,dest_xy=VisualTargets(pixel,destination_pixel).reproject(offset)
    if intervention=='oracle_pixel':
        # Diagnostic only: project the true object-top point, then retain stale calibration.
        truth=env.data.xpos[env.object_id].copy();truth[2]+=.02
        pos=env.data.cam_xpos[env.cam];rotation=env.data.cam_xmat[env.cam].reshape(3,3)
        local=rotation.T@(truth-pos); focal=env.model.cam_fovy[env.cam]
        from environment import FOCAL,IMAGE_W,IMAGE_H
        exact=np.array([IMAGE_W/2+FOCAL*local[0]/(-local[2]),IMAGE_H/2-FOCAL*local[1]/(-local[2])])
        obj=pixel_to_plane(exact)
    elif intervention=='oracle_target':
        obj[:2]=env.data.xpos[env.object_id,:2]  # diagnostic evaluator truth only
    global_model = GlobalCorrection(gp=(method=='gp')) if method in ('global','gp') else None
    grasp_xy = obj[:2] + np.array([-.01161,-.01240])
    result['localization']={'object_pixel':pixel.tolist(),'offset_xy':offset.tolist(),
                            'object_estimate_xyz':obj.tolist(),'grasp_target_xy':grasp_xy.tolist(),
                            'intervention':intervention}
    if integrated_family:
        # Both visual targets are reprojected after the selected calibration.
        result['calibration_events']=calibration_events
        result['localization']['destination_pixel']=destination_pixel.tolist()
        result['localization']['destination_estimate_xy']=dest_xy.tolist()
    if diagnostics:
        result['diagnostic_frames']=[{'image':obs['image'].copy(),'stage':'initial','time_s':obs['time_s'],
                                     'gripper_pixel':None if locate_cyan(obs['image']) is None else locate_cyan(obs['image']).tolist(),
                                     'site_xyz':env.data.site_xpos[env.site].tolist(),
                                     'object_xyz':env.data.xpos[env.object_id].tolist()}]

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
            env.trace[-1]['stage']=name
            env.trace[-1]['target_xyz']=target.tolist()
            env.trace[-1]['predicted_before_xyz']=pos.tolist()
            if env.action_count % 5 == 0: result['frames'].append(obs['image'])
            if diagnostics and 'image' in obs:
                marker=locate_cyan(obs['image']); objpx=locate_green(obs['image'])
                result['diagnostic_frames'].append({'image':obs['image'].copy(),'stage':name,'time_s':obs['time_s'],
                  'target_xyz':target.tolist(),'gripper_pixel':None if marker is None else marker.tolist(),
                  'object_pixel':None if objpx is None else objpx.tolist(),
                  'site_xyz':env.data.site_xpos[env.site].tolist(),'object_xyz':env.data.xpos[env.object_id].tolist()})
        result['stages'][name] = False
        result['last_error_m'] = float(np.linalg.norm(target-pos))
        return False

    if not move('approach', np.r_[grasp_xy,.06], 1.2) or not move('reach', np.r_[grasp_xy,.014], 1.2):
        result.update(success=False, failure='reach')
        return result
    if diagnostics:
        pic=env.observe()['image']; marker=locate_cyan(pic); objpx=locate_green(pic)
        result['diagnostic_frames'].append({'image':pic,'stage':'before close','time_s':float(env.data.time),
          'target_xyz':np.r_[grasp_xy,.014].tolist(),'gripper_pixel':None if marker is None else marker.tolist(),
          'object_pixel':None if objpx is None else objpx.tolist(),'site_xyz':env.data.site_xpos[env.site].tolist(),
          'object_xyz':env.data.xpos[env.object_id].tolist()})
    env.mark_evaluation('before_close')
    for _ in range(15): obs = env.step(np.zeros(5), -.1)
    env.mark_evaluation('after_close')
    if diagnostics:
        pic=env.observe()['image']; marker=locate_cyan(pic); objpx=locate_green(pic)
        result['diagnostic_frames'].append({'image':pic,'stage':'after close','time_s':float(env.data.time),
          'target_xyz':np.r_[grasp_xy,.014].tolist(),'gripper_pixel':None if marker is None else marker.tolist(),
          'object_pixel':None if objpx is None else objpx.tolist(),'site_xyz':env.data.site_xpos[env.site].tolist(),
          'object_xyz':env.data.xpos[env.object_id].tolist()})
    if not move('lift', np.r_[grasp_xy,.06], -.1):
        result.update(success=False, failure='lift_command', evaluation=env.evaluate())
        return result
    env.mark_evaluation('after_lift')
    destination = (np.r_[dest_xy+[-.01161+.01,-.0124+.02],.06] if integrated_family else
                   np.array([.22-.01161+.01,-.05-.0124+.02,.06]))
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
