"""Development-only paired evaluation. Simulator truth stays in this module."""
import numpy as np
from .controllers import JacobianController, ModelController, model_jacobian
from .observations import Camera
from .plant import PlanarPlant


nominal_jacobian = model_jacobian

def reachable(target, camera, lengths=(0.3, 0.2)):
    """Exact radial reachability for the configured full [-pi, pi] joint limits."""
    xy = np.linalg.solve(np.asarray(camera.matrix), np.asarray(target) - camera.offset)
    radius = np.linalg.norm(xy)
    return abs(lengths[0] - lengths[1]) - 1e-12 <= radius <= sum(lengths) + 1e-12


def run(config, adapt=False, method=None):
    plant = PlanarPlant()
    pre = Camera()
    angle = np.deg2rad(config['camera_rotation_deg'])
    post = Camera(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    target = pre.observe(plant.reset(config['target_q_rad']))
    if not reachable(target, post):
        raise ValueError('target unreachable after change')
    plant.reset(config['initial_q_rad'])
    calibration = nominal_jacobian(config['initial_q_rad'])
    if method == 'model':
        controller = ModelController(calibration)
    elif method in (None, 'fixed', 'online'):
        controller = JacobianController(calibration, adapt=adapt if method is None else method == 'online')
    else:
        raise ValueError(f'unknown method: {method}')
    trace = []
    for step in range(config['steps']):
        camera = post if step >= config['change_step'] else pre
        observation = camera.observe(plant.tip_m())
        command = controller.act(observation, target, plant.time_s)
        result = plant.step(command)
        trace.append({'step': step, 'time_s': step * plant.config.step_duration_s,
                      'observation': observation.tolist(), 'target': target.tolist(),
                      'command_rad': command.tolist(), 'executed_rad': result['executed_rad'].tolist(),
                      'controller_saturated': controller.command_saturated, 'plant_saturated': result['saturated'],
                      'error': float(np.linalg.norm(target-observation))})
    errors = np.array([x['error'] for x in trace[config['change_step']:]])
    within = errors <= config['tolerance_camera_units']
    dwell = config['dwell_steps']
    recovery = next((i for i in range(len(within)-dwell+1) if within[i:i+dwell].all()), None)
    summary = {'integrated_normalized_error_s': float(errors.sum()*plant.config.step_duration_s/config['workspace_scale_camera_units']),
               'recovery_time_s': None if recovery is None else (recovery+dwell-1)*plant.config.step_duration_s,
               'recovered': recovery is not None, 'peak_error': float(errors.max()),
               'post_change_actions': len(errors), 'controller_saturated_commands': sum(x['controller_saturated'] for x in trace),
               'plant_saturated_commands': sum(x['plant_saturated'] for x in trace),
               'target_reachable_post_change': True}
    return trace, summary
