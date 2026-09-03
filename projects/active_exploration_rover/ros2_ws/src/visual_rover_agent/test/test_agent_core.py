import json
import math

import pytest

from visual_rover_agent.parser import CommandError, parse_command, serialize_status
from visual_rover_agent.state_machine import ActionMachine, Limits


def parse(payload):
    return parse_command(payload, 0.5, 90.0)


@pytest.mark.parametrize(('payload', 'action', 'value'), [
    ('{"id":"d","action":"drive","distance_m":-0.3}', 'drive', -0.3),
    ('{"id":"t","action":"turn","angle_deg":30}', 'turn', 30.0),
    ('{"id":"s","action":"stop"}', 'stop', 0.0),
])
def test_valid_commands(payload, action, value):
    command = parse(payload)
    assert (command.action, command.value) == (action, value)


@pytest.mark.parametrize(('payload', 'reason'), [
    ('{', 'invalid_json'),
    ('{"id":"x","action":NaN,"distance_m":1}', 'invalid_json'),
    ('{"id":"x"}', 'missing_field'),
    ('{"id":"x","action":"fly"}', 'unknown_action'),
    ('{"id":"x","action":"stop","extra":1}', 'unknown_field'),
    ('{"id":"x","action":"drive","distance_m":true}', 'invalid_number'),
    ('{"id":"x","action":"drive","distance_m":1e999}', 'invalid_number'),
    ('{"id":"x","action":"turn","angle_deg":91}', 'out_of_range'),
])
def test_invalid_commands(payload, reason):
    with pytest.raises(CommandError) as caught:
        parse(payload)
    assert caught.value.reason == reason


def ready_machine(**overrides):
    machine = ActionMachine(Limits(**overrides))
    machine.update_odometry(0.0, 0.0, 0.0, 1.0)
    machine.update_scan(1.0, 1.0, 1.0, 1.0)
    return machine


def test_busy_duplicate_and_stop_preemption_are_zero():
    machine = ready_machine()
    machine.submit(parse('{"id":"one","action":"drive","distance_m":0.3}'), 1.0)
    events, velocity = machine.submit(parse('{"id":"two","action":"turn","angle_deg":20}'), 1.1)
    assert (events[0].state, events[0].reason, velocity) == ('rejected', 'busy', (0.0, 0.0))
    events, velocity = machine.submit(parse('{"id":"one","action":"drive","distance_m":0.2}'), 1.2)
    assert (events[0].reason, velocity) == ('duplicate_active_id', (0.0, 0.0))
    events, velocity = machine.submit(parse('{"id":"stop","action":"stop"}'), 1.3)
    assert [(event.command_id, event.state, event.reason) for event in events] == [
        ('one', 'aborted', 'stopped'), ('stop', 'accepted', ''),
        ('stop', 'succeeded', 'stopped')]
    assert velocity == (0.0, 0.0)


def test_drive_completion_uses_displacement_and_stops():
    machine = ready_machine()
    machine.submit(parse('{"id":"d","action":"drive","distance_m":0.3}'), 1.0)
    machine.update_odometry(0.286, 0.0, 0.0, 1.2)
    events, velocity = machine.tick(1.2)
    assert (events[0].state, velocity) == ('succeeded', (0.0, 0.0))


def test_signed_turn_accumulates_across_wrap_and_completes():
    machine = ready_machine()
    machine.update_odometry(0.0, 0.0, math.radians(170), 1.0)
    machine.submit(parse('{"id":"t","action":"turn","angle_deg":30}'), 1.0)
    machine.update_odometry(0.0, 0.0, math.radians(-171), 1.1)
    assert machine.tick(1.1)[0] == []
    machine.update_odometry(0.0, 0.0, math.radians(-160.5), 1.2)
    events, velocity = machine.tick(1.2)
    assert (events[0].state, velocity) == ('succeeded', (0.0, 0.0))


@pytest.mark.parametrize(('change', 'now', 'reason'), [
    ('obstacle', 1.1, 'obstacle'),
    ('scan', 1.6, 'stale_scan'),
    ('odom', 1.6, 'stale_odometry'),
    ('timeout', 11.1, 'timeout'),
])
def test_abort_paths_stop(change, now, reason):
    limits = {'odometry_staleness_timeout_s': 20.0,
              'scan_staleness_timeout_s': 20.0}
    machine = ready_machine(**limits)
    if change == 'obstacle':
        machine.update_scan(0.3, 1.0, 0.3, 1.0)
    machine.submit(parse('{"id":"d","action":"drive","distance_m":0.3}'), 1.0)
    if change == 'obstacle':
        machine.update_scan(0.2, 1.0, 0.2, now)
        machine.update_odometry(0.0, 0.0, 0.0, now)
    elif change == 'scan':
        machine.limits = Limits(odometry_staleness_timeout_s=20.0)
        machine.update_odometry(0.0, 0.0, 0.0, now)
    elif change == 'odom':
        machine.limits = Limits(scan_staleness_timeout_s=20.0)
        machine.update_scan(1.0, 1.0, 1.0, now)
    else:
        machine.update_scan(1.0, 1.0, 1.0, now)
        machine.update_odometry(0.0, 0.0, 0.0, now)
    events, velocity = machine.tick(now)
    assert (events[0].state, events[0].reason, velocity) == ('aborted', reason, (0.0, 0.0))


def test_missing_or_empty_safety_data_rejects_motion():
    machine = ActionMachine(Limits())
    events, velocity = machine.submit(parse('{"id":"d","action":"drive","distance_m":0.2}'), 1.0)
    assert (events[0].reason, velocity) == ('stale_odometry', (0.0, 0.0))
    machine.update_odometry(0.0, 0.0, 0.0, 1.0)
    machine.update_scan(None, 1.0, 1.0, 1.0)
    assert machine.submit(parse('{"id":"e","action":"drive","distance_m":0.2}'), 1.0)[0][0].reason == 'stale_scan'


def test_status_serialization_is_strict_json():
    assert json.loads(serialize_status('d', 'accepted', '', 12.3456789)) == {
        'id': 'd', 'state': 'accepted', 'reason': '', 'sim_time_s': 12.345679}
