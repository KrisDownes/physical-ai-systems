"""Tests for the shared exploration-result wire contract."""

import json

import pytest

from rover_exploration.result_contract import (
    RESULT_KEYS_V1,
    RESULT_KEYS_V2,
    serialize_result,
    validate_result,
)


def values(**overrides):
    result = {
        'outcome': 'success',
        'blocked_reason': None,
        'completion_time_s': 8.0,
        'goals_assigned': 1,
        'goals_reached': 1,
        'failure_events': 0,
        'temporary_failure_events': 0,
        'permanent_failed_regions': 0,
        'recovery_requests': 0,
        'visited_regions': 1,
        'frontier_cells': 0,
        'frontier_clusters': 0,
        'geometric_frontier_cells': 0,
        'geometric_frontier_clusters': 0,
        'reachable_candidate_clusters': 0,
        'post_exclusion_eligible': 0,
    }
    result.update(overrides)
    return result


def test_current_producer_is_exact_deterministic_v2():
    first = serialize_result(**values())
    second = serialize_result(**values())
    payload = json.loads(first)

    assert first == second
    assert payload['schema_version'] == 2
    assert payload['completed'] is True
    assert set(payload) == set(RESULT_KEYS_V2)
    assert validate_result(payload) == (True, '')


@pytest.mark.parametrize(
    ('change', 'reason'),
    [
        ({'schema_version': True}, 'schema_version is not an integer'),
        ({'outcome': 'unknown'}, 'outcome is not success or blocked'),
        ({'outcome': 'blocked'}, 'blocked_reason is not nonempty text'),
        ({'goals_assigned': True}, 'goals_assigned is not an integer'),
        ({'frontier_cells': -1}, 'frontier_cells is negative'),
    ],
)
def test_invalid_current_values_are_rejected(change, reason):
    payload = {
        'schema_version': 2,
        'completed': True,
        **values(),
        **change,
    }
    valid, detail = validate_result(payload)
    assert not valid
    assert reason in detail


def test_historical_v1_is_accepted_only_with_exact_v1_keys():
    payload = {
        key: value
        for key, value in {
            'schema_version': 1,
            'completed': True,
            **values(),
        }.items()
        if key in RESULT_KEYS_V1
    }
    assert validate_result(payload) == (True, '')

    payload['outcome'] = 'success'
    assert not validate_result(payload)[0]
