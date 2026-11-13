"""Agent-authored, camera-only arm transfer pilot. Not the FLAIR implementation."""
from collections import deque
import time
import numpy as np

from .controllers import JacobianController, model_jacobian
from .plant import PlanarPlant
from .observations import Camera
from .evaluate import reachable

DT = 0.1
MAX_U = 0.1
MAX_DESIRED = 0.025


def nominal_jacobian(obs):
    """Nominal inverse kinematics: identity camera and negative elbow prior."""
    x, y = obs
    l1, l2 = .3, .2
    c = np.clip((x*x + y*y - l1*l1 - l2*l2)/(2*l1*l2), -1, 1)
    q2 = -np.arccos(c)
    q1 = np.arctan2(y, x) - np.arctan2(l2*np.sin(q2), l1+l2*np.cos(q2))
    return model_jacobian((q1, q2))


def desired_step(observation, target):
    d = .7 * (target - observation)
    norm = np.linalg.norm(d)
    return d * min(1., MAX_DESIRED / max(norm, 1e-12))


class TransferController:
    """Shared nominal inverse plus RLS or a command-behavior residual model."""
    def __init__(self, method, initial_observation, forgetting=.95, damping=.02,
                 model_forgetting=.97, gp_length=1., gp_noise=.0015):
        self.method, self.damping = method, damping
        self.online = JacobianController(nominal_jacobian(initial_observation), adapt=True,
                                         forgetting=forgetting, damping=damping)
        self.forgetting, self.gp_length, self.gp_noise = model_forgetting, gp_length, gp_noise
        self.b = np.eye(2)
        self.p = np.eye(2) * 1000.
        self.data = deque(maxlen=24)
        self.previous = None

    def act(self, observation, target, time_s):
        obs = np.asarray(observation)
        if self.method == 'online':
            # The existing controller multiplies target error by 0.7 internally.
            return self.online.act(obs, obs + desired_step(obs, np.asarray(target)) / .7, time_s)
        if self.previous is not None and self.method != 'nominal':
            old_obs, predicted, old_u = self.previous
            delta = obs - old_obs
            # Avoid recording essentially stationary commands; no true state or change flag.
            if np.linalg.norm(old_u) > .003 and np.linalg.norm(predicted) > .0003:
                pz = self.p @ predicted
                k = pz / (self.forgetting + predicted @ pz)
                self.b += np.outer(delta - self.b @ predicted, k)
                self.p = (self.p - np.outer(k, predicted @ self.p)) / self.forgetting
                self.data.append((old_obs.copy(), predicted.copy(), delta.copy()))
        j = nominal_jacobian(obs)
        desired = desired_step(obs, np.asarray(target))
        if self.method in ('global', 'gp'):
            goal = np.linalg.solve(self.b.T @ self.b + .02**2 * np.eye(2), self.b.T @ desired)
            if self.method == 'gp' and len(self.data) >= 4:
                # GP residual depends on observed position and nominal predicted displacement.
                z = np.array([np.r_[p / .20, d / .03] for p, d, _ in self.data])
                ys = np.array([y - self.b @ d for _, d, y in self.data])
                distances = ((z[:, None, :] - z[None, :, :])**2).sum(axis=2)
                kernel = np.exp(-distances / (2*self.gp_length**2))
                query = np.r_[obs / .20, goal / .03]
                cross = np.exp(-((z-query)**2).sum(axis=1)/(2*self.gp_length**2))
                correction = cross @ np.linalg.solve(kernel + self.gp_noise * np.eye(len(z)), ys)
                goal = np.linalg.solve(self.b.T @ self.b + .02**2 * np.eye(2),
                                       self.b.T @ (desired - correction))
        else:
            goal = desired
        u = np.clip(j.T @ np.linalg.solve(j @ j.T + self.damping**2*np.eye(2), goal), -MAX_U, MAX_U)
        self.previous = (obs.copy(), j @ u, u.copy())
        return u


def run_episode(case, method, tuning):
    plant = PlanarPlant()
    pre = Camera()
    angle = np.deg2rad(case.get('rotation_deg', 0.))
    post = Camera(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    plant.reset(case['target_q'])
    target = pre.observe(plant.tip_m())
    pre_q = case.get('pre_target_q', (np.asarray(case['start_q']) + np.asarray(case['target_q'])) / 2)
    plant.reset(pre_q)
    pre_target = pre.observe(plant.tip_m())
    is_reachable = reachable(target, post)
    plant.reset(case['start_q'])
    rng = np.random.default_rng(case['seed'])
    noise = rng.normal(0, case.get('noise', 0.0002), (case['steps']+1, 2))
    initial = pre.observe(plant.tip_m()) + noise[0]
    controller = TransferController(method, initial, **tuning)
    records, computation_s = [], 0.
    for step in range(case['steps']):
        changed = step >= case['change_step']
        camera = post if changed else pre
        current_target = target if changed else pre_target
        obs = camera.observe(plant.tip_m()) + noise[step]
        start = time.perf_counter()
        u = controller.act(obs, current_target, step*DT)
        computation_s += time.perf_counter()-start
        gain = np.asarray(case.get('gain', [1., 1.])) if changed else np.ones(2)
        # Disturbance is evaluator-only; the controller sees the sent command.
        result = plant.step(u * gain)
        records.append({'step': step, 'time_s': step*DT, 'obs': obs.tolist(),
                        'target': current_target.tolist(), 'command_rad': u.tolist(),
                        'executed_rad': result['executed_rad'].tolist(),
                        'error': float(np.linalg.norm(current_target-obs))})
    errors = np.array([r['error'] for r in records[case['change_step']:]])
    tolerance, dwell = .015, 5
    successes = np.convolve((errors <= tolerance).astype(int), np.ones(dwell, int), 'valid')
    hits = np.flatnonzero(successes == dwell)
    recovery = None if len(hits) == 0 else float((hits[0]+dwell-1)*DT)
    positions = np.array([r['obs'] for r in records[case['change_step']:]])
    return records, {'integrated_normalized_error_s': float(errors.sum()*DT/.5),
                     'recovery_time_s': recovery, 'success': recovery is not None,
                     'peak_error': float(errors.max()),
                     'peak_excursion_from_post_change_observation': float(np.linalg.norm(positions - positions[0], axis=1).max()),
                     'action_count': len(records),
                     'post_change_action_count': len(errors), 'computation_wall_s': computation_s,
                     'reachable': bool(is_reachable)}
