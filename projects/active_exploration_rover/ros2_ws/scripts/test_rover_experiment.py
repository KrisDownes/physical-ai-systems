"""Regression checks for independent evaluation, without model or simulation calls."""
import json
from types import SimpleNamespace

from rover_experiment import evaluate


def fixture():
    command = {'id': 'move', 'action': 'drive', 'distance_m': .5}
    monitor = SimpleNamespace(
        poses=[{'t':1.,'x':.4,'y':0.,'yaw':0.},
               {'t':2.,'x':.9,'y':0.,'yaw':0.}],
        commands=[command], statuses=[
            {'id':'move','state':'accepted','sim_time_s':1.},
            {'id':'move','state':'succeeded','sim_time_s':2.}],
        velocity=(0.,0.))
    audit = {'calls':[{'tool':'finish','result':{'content':[{'type':'text','text':json.dumps({
        'finished':True,'stop_result':{'terminal_state':'succeeded'}})}]}}],
        'post_action_ordering':[{'action_id':'move','ordered':True}], 'violations':[]}
    return monitor, audit


def test_success_needs_later_observation_and_finish():
    monitor, audit = fixture()
    assert evaluate(monitor, audit)['success']
    audit['post_action_ordering'] = []
    assert not evaluate(monitor, audit)['success']


def test_finish_transport_success_does_not_imply_stop_success():
    monitor, audit = fixture()
    audit['calls'][0]['result']['content'][0]['text'] = json.dumps({
        'finished':False,'stop_result':{'terminal_state':'aborted'}})
    assert not evaluate(monitor, audit)['success']


def test_motion_measurement_excludes_next_action():
    monitor, audit = fixture()
    monitor.statuses.append({'id':'next','state':'accepted','sim_time_s':2.1})
    monitor.poses.append({'t':2.2,'x':1.5,'y':0.,'yaw':1.})
    result = evaluate(monitor, audit)
    assert result['actions'][0]['translation_m'] == .5
    assert result['actions'][0]['measured_to_sim_s'] == 2.


def test_aborted_action_is_not_navigation_success():
    monitor, audit = fixture()
    monitor.statuses[-1]['state'] = 'aborted'
    assert not evaluate(monitor, audit)['success']


def test_exploration_service_error_is_not_goal_failure_or_success():
    from rover_experiment import exploration_verdict
    result = exploration_verdict({'driver_exit_code': 1},
        {'final_coverage': {'completed': False, 'known_map_percent': 89.3}}, {}, {})
    assert result['termination'] == 'driver_service_error'
    assert not result['success']


def test_coverage_completion_alone_does_not_pass_original_gates():
    from rover_experiment import exploration_verdict
    manifest = {'termination':'coverage_completion','completion_wall_s':400,
                'bag_complete':True,'bag_evidence_complete':True}
    metrics = {'final_coverage':{'completed':True}}
    audit = {'calls':[{}],'violations':[]}
    assert not exploration_verdict(manifest, metrics, {'passed':False}, audit)['success']
    assert exploration_verdict(manifest, metrics, {'passed':True}, audit)['success']
    manifest['completion_wall_s'] = 601
    assert not exploration_verdict(manifest, metrics, {'passed':True}, audit)['success']
