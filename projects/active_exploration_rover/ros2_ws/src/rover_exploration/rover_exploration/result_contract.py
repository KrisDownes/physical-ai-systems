"""Shared exploration-result producer and validator."""

import json
import math


RESULT_KEYS_V1 = (
    'schema_version',
    'completed',
    'completion_time_s',
    'goals_assigned',
    'goals_reached',
    'failure_events',
    'temporary_failure_events',
    'permanent_failed_regions',
    'recovery_requests',
    'visited_regions',
    'frontier_cells',
    'frontier_clusters',
)

RESULT_KEYS_V2 = RESULT_KEYS_V1 + (
    'outcome',
    'blocked_reason',
    'geometric_frontier_cells',
    'geometric_frontier_clusters',
    'reachable_candidate_clusters',
    'post_exclusion_eligible',
)

_COUNTERS_V1 = RESULT_KEYS_V1[3:]
_COUNTERS_V2 = RESULT_KEYS_V2[-4:]


def validate_result(result):
    """Return ``(valid, reason)`` for an exact V1 or V2 result object."""
    if not isinstance(result, dict):
        return False, 'result is not a JSON object'

    version = result.get('schema_version')
    if isinstance(version, bool) or not isinstance(version, int):
        return False, 'result schema_version is not an integer'
    if version == 1:
        required_keys = RESULT_KEYS_V1
    elif version == 2:
        required_keys = RESULT_KEYS_V2
    else:
        return False, 'result schema_version is unsupported'

    if set(result) != set(required_keys):
        return False, 'result keys do not match the required schema version'
    if result['completed'] is not True:
        return False, 'result completed != true'

    completion_time = result['completion_time_s']
    if isinstance(completion_time, bool) or not isinstance(
        completion_time, (int, float)
    ):
        return False, 'completion_time_s is not numeric'
    if not math.isfinite(completion_time) or completion_time < 0.0:
        return False, 'completion_time_s is not a finite non-negative number'

    counters = _COUNTERS_V1
    if version == 2:
        counters += _COUNTERS_V2
    for key in counters:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int):
            return False, f'{key} is not an integer'
        if value < 0:
            return False, f'{key} is negative'

    if version == 2:
        outcome = result['outcome']
        reason = result['blocked_reason']
        if outcome not in ('success', 'blocked'):
            return False, 'result outcome is not success or blocked'
        if outcome == 'success' and reason is not None:
            return False, 'successful result blocked_reason is not null'
        if outcome == 'blocked' and (
            not isinstance(reason, str) or not reason.strip()
        ):
            return False, 'blocked result blocked_reason is not nonempty text'

    return True, ''


def build_result(**values):
    """Build and validate the one current producer payload (schema V2)."""
    result = {**values, 'schema_version': 2, 'completed': True}
    valid, reason = validate_result(result)
    if not valid:
        raise ValueError(reason)
    return result


def serialize_result(**values):
    """Return deterministic JSON for a valid current result."""
    return json.dumps(build_result(**values), sort_keys=True)
