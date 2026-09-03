"""Strict parsing for the deliberately small agent command language."""

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """A validated semantic command."""

    command_id: str
    action: str
    value: float = 0.0


class CommandError(ValueError):
    """A command rejection with its stable reason code."""

    def __init__(self, reason, command_id=''):
        """Initialize the rejection."""
        super().__init__(reason)
        self.reason = reason
        self.command_id = command_id


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CommandError('unknown_field')
        result[key] = value
    return result


def parse_command(
        payload, maximum_drive_distance_m, maximum_turn_angle_deg):
    """Parse strict JSON, rejecting extensions and non-finite JSON numbers."""
    try:
        data = json.loads(
            payload,
            object_pairs_hook=_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                CommandError('invalid_json')
            ),
        )
    except CommandError:
        raise
    except (json.JSONDecodeError, TypeError):
        raise CommandError('invalid_json') from None
    if not isinstance(data, dict):
        raise CommandError('invalid_json')

    command_id = data.get('id', '')
    if not isinstance(command_id, str) or not command_id:
        raise CommandError('missing_field')
    if 'action' not in data:
        raise CommandError('missing_field', command_id)
    action = data['action']
    if action not in ('drive', 'turn', 'stop'):
        raise CommandError('unknown_action', command_id)

    expected = {'id', 'action'}
    field = None
    limit = None
    if action == 'drive':
        field, limit = 'distance_m', maximum_drive_distance_m
        expected.add(field)
    elif action == 'turn':
        field, limit = 'angle_deg', maximum_turn_angle_deg
        expected.add(field)
    if set(data) != expected:
        reason = (
            'missing_field'
            if not expected.issubset(data)
            else 'unknown_field'
        )
        raise CommandError(reason, command_id)
    if action == 'stop':
        return Command(command_id, action)

    value = data[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommandError('invalid_number', command_id)
    value = float(value)
    if not math.isfinite(value):
        raise CommandError('invalid_number', command_id)
    if value == 0.0 or abs(value) > limit:
        raise CommandError('out_of_range', command_id)
    return Command(command_id, action, value)


def serialize_status(command_id, state, reason, sim_time_s):
    """Produce compact strict JSON with a stable field set."""
    return json.dumps({
        'id': command_id,
        'state': state,
        'reason': reason,
        'sim_time_s': round(sim_time_s, 6),
    }, separators=(',', ':'), allow_nan=False)
