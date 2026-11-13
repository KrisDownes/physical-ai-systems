"""Checks of the development experiment's behavioral boundaries."""
import json
from pathlib import Path
import numpy as np
from active_adaptation.evaluate import nominal_jacobian, reachable, run
from active_adaptation.observations import Camera
from active_adaptation.plant import PlanarPlant


def config():
    return json.loads(Path('configs/development.json').read_text())


def test_jacobian_matches_small_command_response():
    p = PlanarPlant()
    q = (0.3, -1.3)
    initial = p.reset(q)
    columns = []
    for action in ((1e-6, 0), (0, 1e-6)):
        p.reset(q)
        columns.append((p.step(action)['tip_m'] - initial) / 1e-6)
    np.testing.assert_allclose(np.column_stack(columns), nominal_jacobian(q), atol=3e-7)


def test_repeatability_change_and_limits():
    c = config()
    trace, summary = run(c, True)
    assert (trace, summary) == run(c, True)
    assert all(np.max(np.abs(row['command_rad'])) <= 0.1 for row in trace)
    no_change = dict(c, camera_rotation_deg=0)
    reference, _ = run(no_change, True)
    assert trace[:c['change_step']] == reference[:c['change_step']]
    assert trace[c['change_step']]['observation'] != reference[c['change_step']]['observation']
    assert not reachable((1.0, 0.0), Camera())


def test_unrecovered_remains_in_summary():
    c = dict(config(), camera_rotation_deg=150)
    trace, summary = run(c, False)
    assert len(trace) == c['steps']
    assert not summary['recovered']
    assert summary['recovery_time_s'] is None
    assert summary['integrated_normalized_error_s'] > 0


def test_model_controller_infers_initial_configuration_and_updates_from_commands():
    from active_adaptation.controllers import ModelController
    p = PlanarPlant()
    q = np.array((0.3, -1.3))
    obs = p.reset(q)
    controller = ModelController(nominal_jacobian(q))
    command = controller.act(obs, np.array((0.3, 0.1)), 0.0)
    np.testing.assert_allclose(controller.q_estimate, q, atol=1e-12)
    result = p.step(command)
    controller.act(result['tip_m'], np.array((0.3, 0.1)), 0.1)
    np.testing.assert_allclose(controller.q_estimate, p.q, atol=1e-12)


def test_model_baseline_repeats_and_retains_failures():
    c = config()
    trace, summary = run(c, method='model')
    assert (trace, summary) == run(c, method='model')
    assert len(trace) == c['steps']
    assert not summary['recovered']
    assert summary['recovery_time_s'] is None
