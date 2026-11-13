import numpy as np
from active_adaptation.arm_transfer import TransferController, desired_step, run_episode


def test_desired_motion_is_bounded_and_commands_are_limited():
    obs, target = np.array([.3, 0]), np.array([3., 2.])
    assert np.linalg.norm(desired_step(obs, target)) <= .025 + 1e-12
    for method in ('nominal', 'online', 'global', 'gp'):
        u = TransferController(method, obs).act(obs, target, 0.)
        assert np.all(np.abs(u) <= .1 + 1e-12)


def test_paired_noise_and_initial_observations_reproduce():
    case = {'start_q': [0., -1.], 'target_q': [.4, -1.4], 'seed': 42,
            'steps': 30, 'change_step': 10, 'gain': [.5, 1.]}
    a, _ = run_episode(case, 'nominal', {})
    b, _ = run_episode(case, 'nominal', {})
    c, _ = run_episode(case, 'global', {})
    assert a == b
    assert a[0]['obs'] == c[0]['obs']
    assert a[0]['target'] == c[0]['target']


def test_same_tip_different_elbow_has_different_joint_jacobian():
    from active_adaptation.plant import PlanarPlant
    from active_adaptation.controllers import model_jacobian
    p = PlanarPlant()
    x = p.reset([0., 1.])
    q2 = -1.
    q1 = np.arctan2(x[1], x[0]) - np.arctan2(.2*np.sin(q2), .3+.2*np.cos(q2))
    assert np.allclose(p.reset([q1, q2]), x)
    assert not np.allclose(model_jacobian([0., 1.]), model_jacobian([q1, q2]))
