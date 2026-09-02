"""ROS-adapter tests for terminal publication and latching."""

from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSReliabilityPolicy

from rover_exploration.exploration_policy import (
    ExplorationPolicy,
    PolicyConfig,
)
from rover_exploration.frontier_node import FrontierDetector
from rover_exploration.result_contract import RESULT_KEYS_V2, validate_result


def config():
    return PolicyConfig(
        0.25, 3, 3, 6.0, 0.05, 0.39, 0.75, 30.0, 2, 0.60,
        0.20, 2.0, 8.0, 1.5,
    )


def terminal_policy():
    policy = ExplorationPolicy(config())
    values = {
        'raw_data': [0],
        'planning_data': [0],
        'width': 1,
        'height': 1,
        'resolution': 1.0,
        'origin_x': 0.0,
        'origin_y': 0.0,
        'frontier_cells': set(),
        'frontier_clusters': [],
        'robot_cell': (0, 0),
        'robot_world': (0.5, 0.5),
    }
    policy.update(**values, now_s=0.0)
    assert policy.update(**values, now_s=8.0).completed_now
    return policy


def test_terminal_publishes_exact_result_once():
    node = FrontierDetector.__new__(FrontierDetector)
    node.policy = terminal_policy()
    node.terminal_published = False
    node.node_time_s = lambda: 8.0
    node.exploration_complete_publisher = MagicMock()
    node.exploration_result_publisher = MagicMock()
    node.get_logger = MagicMock(return_value=MagicMock())

    node._publish_terminal()
    node._publish_terminal()

    state = node.exploration_complete_publisher.publish.call_args.args[0]
    result = node.exploration_result_publisher.publish.call_args.args[0]
    payload = json.loads(result.data)
    assert state.data is True
    assert set(payload) == set(RESULT_KEYS_V2)
    assert payload['outcome'] == 'success'
    assert node.exploration_complete_publisher.publish.call_count == 1
    assert node.exploration_result_publisher.publish.call_count == 1


def test_terminal_map_callback_only_publishes_empty_path():
    node = FrontierDetector.__new__(FrontierDetector)
    node.policy = terminal_policy()
    node._publish_path = MagicMock()
    message = SimpleNamespace(header=object(), info=object())

    node.map_callback(message)

    node._publish_path.assert_called_once_with(message.header, message.info, None)


def test_empty_path_precedes_terminal_publication():
    node = FrontierDetector.__new__(FrontierDetector)
    node.policy = ExplorationPolicy(config())
    policy_values = {
        'raw_data': [0, -1],
        'planning_data': [0, -1],
        'width': 2,
        'height': 1,
        'resolution': 1.0,
        'origin_x': 0.0,
        'origin_y': 0.0,
        'frontier_cells': {(0, 0)},
        'frontier_clusters': [],
        'robot_cell': (0, 0),
        'robot_world': (0.5, 0.5),
    }
    assert node.policy.update(**policy_values, now_s=0.0).debounce_started
    node.node_time_s = lambda: 8.0
    node._map_pose = MagicMock(return_value=((0.5, 0.5), (0, 0)))
    node._build_planning_grid = MagicMock(return_value=([0, -1], 1))

    calls = MagicMock()
    node._publish_grid = calls.grid
    node._publish_path = calls.path
    node._publish_markers = calls.markers
    node._log_cycle = MagicMock()
    node._log_transition = MagicMock()
    node._publish_terminal = calls.terminal
    info = SimpleNamespace(
        width=2,
        height=1,
        resolution=1.0,
        origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0)),
    )
    message = SimpleNamespace(header=object(), info=info, data=[0, -1])

    node.map_callback(message)

    names = [call[0] for call in calls.mock_calls]
    assert names.index('path') < names.index('terminal')
    assert calls.path.call_args.args[2] is None
    assert calls.markers.call_args.args[1] == set()


def test_completed_policy_cannot_request_recovery():
    node = FrontierDetector.__new__(FrontierDetector)
    node.policy = terminal_policy()
    node.latest_pose = (0.0, 0.0, 0.0)
    node.node_time_s = lambda: 20.0
    node.recovery_request_publisher = MagicMock()

    node.stuck_check_callback()

    node.recovery_request_publisher.publish.assert_not_called()


def test_recovery_lifecycle_suppresses_duplicates_and_rearms():
    policy = ExplorationPolicy(config())
    node = FrontierDetector.__new__(FrontierDetector)
    node.policy = policy
    node.latest_pose = (1.5, 5.5, 0.0)
    node.recovery_request_publisher = MagicMock()
    node.get_logger = MagicMock(return_value=MagicMock())

    def plan(cluster, now_s):
        width, height = 21, 11
        data = [0] * (width * height)
        return policy.update(
            raw_data=data,
            planning_data=list(data),
            width=width,
            height=height,
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            frontier_cells=set(cluster),
            frontier_clusters=[set(cluster)],
            robot_cell=(5, 1),
            robot_world=(1.5, 5.5),
            now_s=now_s,
        )

    first = {(5, column) for column in range(10, 15)}
    assert plan(first, 0.0).path
    times = iter((0.0, 5.0, 6.0, 10.0, 15.0))
    node.node_time_s = lambda: next(times)

    node.stuck_check_callback()
    node.stuck_check_callback()
    node.stuck_check_callback()
    assert node.recovery_request_publisher.publish.call_count == 1
    assert policy.counters.recovery_requests == 1

    node.recovery_status_callback(SimpleNamespace(data=True))
    assert plan({(8, column) for column in range(4, 9)}, 7.0).path is None
    node.recovery_status_callback(SimpleNamespace(data=False))
    second = {(8, column) for column in range(4, 9)}
    assert plan(second, 8.0).path

    node.stuck_check_callback()
    node.stuck_check_callback()
    assert node.recovery_request_publisher.publish.call_count == 2
    assert policy.counters.recovery_requests == 2

    node.recovery_status_callback(SimpleNamespace(data=True))
    node.recovery_status_callback(SimpleNamespace(data=False))
    third = {(2, column) for column in range(15, 20)}
    assert plan(third, 16.0).path


def test_blocked_policy_publishes_valid_strict_v2_result_once():
    blocked_config = replace(config(), goal_reached_distance_m=100.0)
    policy = ExplorationPolicy(blocked_config)
    cluster = {(0, column) for column in range(5)}
    values = {
        'raw_data': [0] * 6,
        'planning_data': [0] * 6,
        'width': 6,
        'height': 1,
        'resolution': 1.0,
        'origin_x': 0.0,
        'origin_y': 0.0,
        'frontier_cells': cluster,
        'frontier_clusters': [cluster],
        'robot_cell': (0, 0),
        'robot_world': (0.5, 0.5),
    }
    assert policy.update(**values, now_s=0.0).debounce_started
    assert policy.update(**values, now_s=8.0).completed_now
    assert policy.terminal.outcome == 'blocked'

    node = FrontierDetector.__new__(FrontierDetector)
    node.policy = policy
    node.terminal_published = False
    node.node_time_s = lambda: 8.0
    node.exploration_complete_publisher = MagicMock()
    node.exploration_result_publisher = MagicMock()
    node.get_logger = MagicMock(return_value=MagicMock())

    node._publish_terminal()
    node._publish_terminal()

    result = node.exploration_result_publisher.publish.call_args.args[0]
    payload = json.loads(result.data)
    assert validate_result(payload) == (True, '')
    assert set(payload) == set(RESULT_KEYS_V2)
    assert payload['outcome'] == 'blocked'
    assert payload['blocked_reason']
    assert node.exploration_complete_publisher.publish.call_count == 1
    assert node.exploration_result_publisher.publish.call_count == 1


def test_completion_topics_use_latched_reliable_qos_and_initial_false():
    publishers = {}

    def create_publisher(_self, _type, topic, qos):
        publisher = MagicMock()
        publishers[topic] = (publisher, qos)
        return publisher

    with (
        patch.object(Node, '__init__', return_value=None),
        patch.object(FrontierDetector, '_parameter', side_effect=lambda _, default: default),
        patch.object(FrontierDetector, 'create_publisher', new=create_publisher),
        patch.object(FrontierDetector, 'create_subscription', return_value=MagicMock()),
        patch.object(FrontierDetector, 'create_timer', return_value=MagicMock()),
        patch('rover_exploration.frontier_node.Buffer', return_value=MagicMock()),
        patch('rover_exploration.frontier_node.TransformListener', return_value=MagicMock()),
    ):
        FrontierDetector()

    for topic in ('/exploration_complete', '/exploration_result'):
        qos = publishers[topic][1]
        assert qos.reliability == QoSReliabilityPolicy.RELIABLE
        assert qos.durability == QoSDurabilityPolicy.TRANSIENT_LOCAL
        assert qos.depth == 1
    initial = publishers['/exploration_complete'][0].publish.call_args.args[0]
    assert initial.data is False
