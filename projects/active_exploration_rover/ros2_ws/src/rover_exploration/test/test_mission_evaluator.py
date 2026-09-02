"""
Test the V15 mission evaluator (receipt-time / simulation-time fixes).

Bag reading is exercised separately; here we test the pure metric, clock
mapping, completion-validation, and pass/fail logic directly with synthetic
ROS messages so no large bag is required.
"""

import math

from rover_exploration import mission_evaluator as me


def make_bool(value):
    from std_msgs.msg import Bool
    m = Bool()
    m.data = value
    return m


def make_string(text):
    from std_msgs.msg import String
    m = String()
    m.data = text
    return m


def make_clock(receipt_ns, sim_s):
    from rosgraph_msgs.msg import Clock
    m = Clock()
    m.clock.sec = int(sim_s)
    m.clock.nanosec = int((sim_s - int(sim_s)) * 1e9)
    return receipt_ns, m


def make_map(percent_known=99.0, size=10):
    from nav_msgs.msg import OccupancyGrid
    m = OccupancyGrid()
    m.info.width = size
    m.info.height = size
    known = int(percent_known / 100.0 * (size * size))
    m.data = [0] * known + [-1] * (size * size - known)
    return m


def make_odom(receipt_ns, sim_s, x, y, yaw, frame='odom'):
    from geometry_msgs.msg import Quaternion
    from nav_msgs.msg import Odometry
    m = Odometry()
    m.header.stamp.sec = int(sim_s)
    m.header.stamp.nanosec = int((sim_s - int(sim_s)) * 1e9)
    m.header.frame_id = frame
    m.pose.pose.position.x = x
    m.pose.pose.position.y = y
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    m.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
    return receipt_ns, m


def make_tf(receipt_ns, sim_s, tx, ty, yaw, frame='map', child='odom'):
    from geometry_msgs.msg import Quaternion, TransformStamped
    from geometry_msgs.msg import Vector3
    from tf2_msgs.msg import TFMessage
    tr = TransformStamped()
    tr.header.stamp.sec = int(sim_s)
    tr.header.stamp.nanosec = int((sim_s - int(sim_s)) * 1e9)
    tr.header.frame_id = frame
    tr.child_frame_id = child
    tr.transform.translation = Vector3(x=tx, y=ty, z=0.0)
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    tr.transform.rotation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
    msg = TFMessage()
    msg.transforms.append(tr)
    return receipt_ns, msg


def make_path(receipt_ns, poses):
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    p = Path()
    for _ in range(poses):
        p.poses.append(PoseStamped())
    return receipt_ns, p


def make_twist(x=0.0, z=0.0):
    from geometry_msgs.msg import Twist
    t = Twist()
    t.linear.x = x
    t.angular.z = z
    return t


def result_payload(**overrides):
    payload = {
        'schema_version': 1,
        'completed': True,
        'completion_time_s': 100.0,
        'goals_assigned': 3,
        'goals_reached': 3,
        'failure_events': 0,
        'temporary_failure_events': 0,
        'permanent_failed_regions': 0,
        'recovery_requests': 0,
        'visited_regions': 3,
        'frontier_cells': 0,
        'frontier_clusters': 0,
    }
    payload.update(overrides)
    return payload


def result_payload_v2(**overrides):
    payload = result_payload(
        schema_version=2,
        outcome='success',
        blocked_reason=None,
        geometric_frontier_cells=0,
        geometric_frontier_clusters=0,
        reachable_candidate_clusters=0,
        post_exclusion_eligible=0,
    )
    payload.update(overrides)
    return payload


# Realistic receipt timestamps (Unix epoch ~1.787e18 ns) while /clock carries
# small simulation seconds (0-150). This exposes receipt/sim-time mixing.
RECEIPT_BASE = 1_787_000_000_000_000_000


def build_successful_mission():
    """
    Build a passing mission with realistic receipt/sim time separation.

    Bag receipt timestamps use the Unix epoch (~1.787e18 ns) while /clock
    carries small simulation seconds (0-150), exposing any receipt/sim
    time mixing in the evaluator.
    """
    collected = {topic: [] for topic in me.REQUIRED_TOPICS}
    # /clock: receipt ns maps to small sim seconds via the mapper.
    collected['/clock'] = [
        make_clock(RECEIPT_BASE + 0, 0.0),
        make_clock(RECEIPT_BASE + 10_000_000_000, 150.0),
    ]
    # Completion: false then true, with realistic receipt timestamps.
    collected['/exploration_complete'] = [
        (RECEIPT_BASE + 1_000_000_000, make_bool(False)),
        (RECEIPT_BASE + 5_000_000_000, make_bool(True)),
    ]
    # The completion receipt (5e9) maps through the /clock pair
    # (0->0, 1e10->150) to sim time 75.0, so the result must carry the
    # matching completion_time_s to be associated by evaluate_mission.
    collected['/exploration_result'] = [
        (
            RECEIPT_BASE + 5_000_000_001,
            make_string(
                __import__('json').dumps(
                    result_payload(completion_time_s=75.0)
                )
            ),
        )
    ]
    collected['/map'] = [(RECEIPT_BASE + 2_000_000_000, make_map(99.0))]
    collected['/cmd_vel'] = [
        (RECEIPT_BASE + 3_000_000_000, make_twist(0.0, 0.0))
    ]
    collected['/cmd_vel_raw'] = [
        (RECEIPT_BASE + 3_000_000_000, make_twist(0.0, 0.0))
    ]
    collected['/planned_path'] = [
        make_path(RECEIPT_BASE + 3_000_000_000, 0)
    ]
    # Stationary (filtered and ground truth agree, rover stopped).
    # Two samples straddle completion (at sim 80 and 140) so the
    # after-completion ground-truth window has >= 2 samples.
    collected['/odometry/filtered'] = [
        make_odom(RECEIPT_BASE + 3_000_000_000, 80.0, 0.0, 0.0, 0.0),
        make_odom(RECEIPT_BASE + 6_000_000_000, 140.0, 0.0, 0.0, 0.0),
    ]
    collected['/ground_truth/odometry'] = [
        make_odom(
            RECEIPT_BASE + 3_000_000_000, 80.0, 0.0, 0.0, 0.0,
            frame='base_footprint',
        ),
        make_odom(
            RECEIPT_BASE + 5_500_000_000, 110.0, 0.0, 0.0, 0.0,
            frame='base_footprint',
        ),
        make_odom(
            RECEIPT_BASE + 6_000_000_000, 140.0, 0.0, 0.0, 0.0,
            frame='base_footprint',
        ),
    ]
    # Two map->odom transforms (>= 2 required) so the correction-step
    # telemetry check passes.
    collected['/tf'] = [
        make_tf(RECEIPT_BASE + 3_000_000_000, 80.0, 0.01, 0.0, 0.01),
        make_tf(RECEIPT_BASE + 6_000_000_000, 140.0, 0.02, 0.0, 0.02),
    ]
    collected['/recovery_request'] = []
    return collected


# ---------------------------------------------------------------------------
# Clock mapping (receipt ns -> simulation seconds)
# ---------------------------------------------------------------------------

def test_receipt_to_sim_clamps_below_and_above():
    pairs = [
        (RECEIPT_BASE + 0, 0.0),
        (RECEIPT_BASE + 10_000_000_000, 100.0),
        (RECEIPT_BASE + 20_000_000_000, 200.0),
    ]
    assert me.receipt_to_sim(RECEIPT_BASE - 5, pairs) == 0.0
    assert me.receipt_to_sim(RECEIPT_BASE + 99_000_000_000, pairs) == 200.0


def test_receipt_to_sim_interpolates_linearly():
    pairs = [
        (RECEIPT_BASE + 0, 0.0),
        (RECEIPT_BASE + 10_000_000_000, 100.0),
    ]
    got = me.receipt_to_sim(RECEIPT_BASE + 5_000_000_000, pairs)
    assert abs(got - 50.0) < 1e-6


def test_receipt_to_sim_never_mixes_domains():
    # Realistic receipt ns with tiny sim values: a direct subtraction would
    # give ~ -1.787e9. The mapper must return the small sim value.
    pairs = [
        (RECEIPT_BASE + 0, 0.0),
        (RECEIPT_BASE + 10_000_000_000, 150.0),
    ]
    got = me.receipt_to_sim(RECEIPT_BASE + 5_000_000_000, pairs)
    assert 0.0 < got < 200.0


# ---------------------------------------------------------------------------
# Completion time + final-state validation
# ---------------------------------------------------------------------------

def test_completion_time_s_is_nonzero_and_correct():
    collected = build_successful_mission()
    result, _, _ = me.evaluate_mission(collected)
    # Completing transition receipt ns = RECEIPT_BASE + 5e9 maps between
    # clock(0 -> 0s) and clock(10e9 -> 150s): 75s.
    assert abs(result['completion_time_s'] - 75.0) < 1e-6
    assert result['completion_time_s'] > 0.0


def test_post_completion_duration_uses_sim_domain():
    collected = build_successful_mission()
    result, _, _ = me.evaluate_mission(collected)
    # final clock sim = 150; completion sim = 75 => 75s observation.
    assert abs(result['post_completion_observation_s'] - 75.0) < 1e-6


def test_false_true_false_fails_final_state_false():
    collected = build_successful_mission()
    collected['/exploration_complete'].append(
        (RECEIPT_BASE + 7_000_000_000, make_bool(False))
    )
    collected['/exploration_result'] = []  # no result after reversion
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('final' in r.lower() for r in reasons)


def test_true_false_true_fails_first_state_not_false():
    # true -> false -> true: a passing final state, but the FIRST recorded
    # state was not false, so it must fail on that rule specifically.
    collected = build_successful_mission()
    collected['/exploration_complete'] = [
        (RECEIPT_BASE + 1_000_000_000, make_bool(True)),
        (RECEIPT_BASE + 3_000_000_000, make_bool(False)),
        (RECEIPT_BASE + 5_000_000_000, make_bool(True)),
    ]
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 5_000_000_001, make_string(
            __import__('json').dumps(result_payload())))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('first recorded' in r.lower() for r in reasons)


def test_zero_transitions_fails():
    collected = build_successful_mission()
    # Only a single false (no transition at all).
    collected['/exploration_complete'] = [
        (RECEIPT_BASE + 1_000_000_000, make_bool(False))
    ]
    collected['/exploration_result'] = []
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('transition' in r.lower() for r in reasons)


def test_multiple_transitions_fails():
    collected = build_successful_mission()
    # false -> true -> false -> true : two false->true transitions.
    collected['/exploration_complete'] = [
        (RECEIPT_BASE + 1_000_000_000, make_bool(False)),
        (RECEIPT_BASE + 3_000_000_000, make_bool(True)),
        (RECEIPT_BASE + 5_000_000_000, make_bool(False)),
        (RECEIPT_BASE + 7_000_000_000, make_bool(True)),
    ]
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 7_000_000_001, make_string(
            __import__('json').dumps(result_payload())))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('one false' in r.lower() for r in reasons)


def test_true_without_recorded_false_fails():
    collected = build_successful_mission()
    collected['/exploration_complete'] = [
        (RECEIPT_BASE + 5_000_000_000, make_bool(True))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('first recorded' in r.lower() for r in reasons)


def test_synthetic_successful_mission_passes():
    collected = build_successful_mission()
    result, passed, reasons = me.evaluate_mission(collected)
    assert passed, reasons


# ---------------------------------------------------------------------------
# Result validation
# ---------------------------------------------------------------------------

def test_empty_result_string_fails():
    collected = build_successful_mission()
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 5_000_000_001, make_string(''))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('result' in r.lower() for r in reasons)


def test_malformed_json_result_fails():
    collected = build_successful_mission()
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 5_000_000_001, make_string('{not json'))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed


def test_missing_key_result_fails():
    collected = build_successful_mission()
    payload = result_payload()
    del payload['goals_assigned']
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 5_000_000_001, make_string(__import__('json').dumps(payload)))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed


def test_extra_key_result_fails():
    collected = build_successful_mission()
    payload = result_payload()
    payload['bonus'] = 1
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 5_000_000_001, make_string(__import__('json').dumps(payload)))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed


def test_wrong_schema_version_fails():
    collected = build_successful_mission()
    payload = result_payload(schema_version=2)
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 5_000_000_001, make_string(__import__('json').dumps(payload)))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed


def test_legacy_v1_result_is_accepted_with_implicit_success():
    result, passed, _ = me.evaluate_mission(build_successful_mission())
    assert passed
    assert result['result_outcome'] == 'success'
    assert result['blocked_reason'] is None


def test_v2_success_result_is_accepted():
    collected = build_successful_mission()
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 5_000_000_001, make_string(
            __import__('json').dumps(result_payload_v2(
                completion_time_s=75.0))))
    ]
    result, passed, _ = me.evaluate_mission(collected)
    assert passed
    assert result['result_outcome'] == 'success'


def test_v2_blocked_result_propagates_reason():
    collected = build_successful_mission()
    reason = 'fresh approach retry cap exhausted'
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 5_000_000_001, make_string(
            __import__('json').dumps(result_payload_v2(
                completion_time_s=75.0, outcome='blocked',
                blocked_reason=reason))))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert result['passed'] is False
    assert result['result_outcome'] == 'blocked'
    assert result['blocked_reason'] == reason
    assert any(reason in item for item in reasons)


def test_v1_payload_with_v2_keys_is_rejected():
    collected = build_successful_mission()
    payload = result_payload(outcome='success', blocked_reason=None)
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 5_000_000_001, make_string(
            __import__('json').dumps(payload)))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert result['result_outcome'] is None
    assert any('keys' in reason for reason in reasons)


def test_invalid_result_never_defaults_outcome_to_success():
    collected = build_successful_mission()
    _set_result(collected, schema_version=99)
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert result['result_outcome'] is None
    assert any('schema' in reason for reason in reasons)


def test_boolean_schema_version_is_rejected():
    collected = build_successful_mission()
    _set_result(collected, schema_version=True)
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert result['result_outcome'] is None
    assert any('schema_version is not an integer' in reason for reason in reasons)


def test_float_schema_version_is_rejected():
    collected = build_successful_mission()
    _set_result(collected, schema_version=2.0)
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert result['result_outcome'] is None
    assert any('schema_version is not an integer' in reason for reason in reasons)


def test_v2_counter_with_boolean_type_is_rejected():
    collected = build_successful_mission()
    payload = result_payload_v2(
        completion_time_s=75.0,
        geometric_frontier_cells=True,
    )
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 5_000_000_001, make_string(
            __import__('json').dumps(payload)))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert result['result_outcome'] is None
    assert any(
        'geometric_frontier_cells is not an integer' in reason
        for reason in reasons
    )


def test_completed_false_result_fails():
    collected = build_successful_mission()
    payload = result_payload(completed=False)
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 5_000_000_001, make_string(__import__('json').dumps(payload)))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed


# ---------------------------------------------------------------------------
# Strengthened structured-result validation
# ---------------------------------------------------------------------------

def _set_result(collected, **overrides):
    payload = result_payload(**overrides)
    collected['/exploration_result'] = [
        (RECEIPT_BASE + 5_000_000_001, make_string(
            __import__('json').dumps(payload)))
    ]


def test_result_boolean_completion_time_fails():
    collected = build_successful_mission()
    _set_result(collected, completion_time_s=True)
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed


def test_result_nan_completion_time_fails():
    collected = build_successful_mission()
    _set_result(collected, completion_time_s=float('nan'))
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed


def test_result_inf_completion_time_fails():
    collected = build_successful_mission()
    _set_result(collected, completion_time_s=float('inf'))
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed


def test_result_negative_completion_time_fails():
    collected = build_successful_mission()
    _set_result(collected, completion_time_s=-1.0)
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed


def test_result_boolean_counter_fails():
    collected = build_successful_mission()
    _set_result(collected, goals_assigned=True)
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed


def test_result_negative_counter_fails():
    collected = build_successful_mission()
    _set_result(collected, recovery_requests=-1)
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed


def test_result_timestamp_inconsistent_with_transition_fails():
    # Result completion_time_s must agree with the mapped transition time
    # (75s in build_successful_mission) within 0.5s.
    collected = build_successful_mission()
    _set_result(collected, completion_time_s=200.0)
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('result' in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# Telemetry sufficiency
# ---------------------------------------------------------------------------

def test_insufficient_clock_messages_fails():
    collected = build_successful_mission()
    collected['/clock'] = [make_clock(RECEIPT_BASE + 0, 0.0)]  # only one
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('clock' in r.lower() for r in reasons)


def test_no_final_map_fails():
    collected = build_successful_mission()
    collected['/map'] = []
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('map' in r.lower() for r in reasons)


def test_insufficient_filtered_odometry_fails():
    collected = build_successful_mission()
    collected['/odometry/filtered'] = [
        make_odom(RECEIPT_BASE + 3_000_000_000, 80.0, 0.0, 0.0, 0.0)
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('odometry/filtered' in r.lower() for r in reasons)


def test_insufficient_ground_truth_fails():
    collected = build_successful_mission()
    collected['/ground_truth/odometry'] = [
        make_odom(RECEIPT_BASE + 3_000_000_000, 80.0, 0.0, 0.0, 0.0,
                  frame='base_footprint')
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('ground_truth' in r.lower() for r in reasons)


def test_insufficient_map_to_odom_transforms_fails():
    collected = build_successful_mission()
    collected['/tf'] = [
        make_tf(RECEIPT_BASE + 3_000_000_000, 80.0, 0.01, 0.0, 0.01)
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('map->odom' in r.lower() for r in reasons)


def test_insufficient_ground_truth_after_completion_fails():
    collected = build_successful_mission()
    # Only one ground-truth sample after completion.
    collected['/ground_truth/odometry'] = [
        make_odom(RECEIPT_BASE + 6_000_000_000, 140.0, 0.0, 0.0, 0.0,
                  frame='base_footprint')
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('after completion' in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# Frontier component reporting (residual components + selectable count)
# ---------------------------------------------------------------------------

def test_residual_frontier_components_reported_selectable_zero():
    # A map that is >= 98% known but contains small residual frontier
    # clusters (each < 5 cells), so selectable count stays zero.
    from nav_msgs.msg import OccupancyGrid
    collected = build_successful_mission()
    size = 15
    m = OccupancyGrid()
    m.info.width = size
    m.info.height = size
    # All free (known), then carve a few isolated unknown pockets.
    data = [0] * (size * size)
    # Three separate single unknown cells (interior) -> each yields a
    # small (<=4 cell) frontier cluster around it.
    for idx in (11, 44, 77):
        data[idx] = -1
    m.data = data
    collected['/map'] = [(RECEIPT_BASE + 2_000_000_000, m)]
    result, passed, reasons = me.evaluate_mission(collected)
    # All residual sizes reported, none >= 5 -> selectable == 0.
    assert result['final_frontier_component_sizes']
    assert all(s < 5 for s in result['final_frontier_component_sizes'])
    assert result['selectable_frontier_components'] == 0
    assert passed, reasons


def test_five_cell_component_fails():
    from nav_msgs.msg import OccupancyGrid
    collected = build_successful_mission()
    m = OccupancyGrid()
    size = 9
    m.info.width = size
    m.info.height = size
    data = [-1] * (size * size)
    # A horizontal run of 5 free cells (one cluster >= 5).
    for i in range(5):
        data[i] = 0
    m.data = data
    collected['/map'] = [(RECEIPT_BASE + 2_000_000_000, m)]
    result, passed, reasons = me.evaluate_mission(collected)
    assert any(s >= 5 for s in result['final_frontier_component_sizes'])
    assert not passed
    assert any('>= 5' in r for r in reasons)


# ---------------------------------------------------------------------------
# Post-completion displacement (max excursion, not endpoint delta)
# ---------------------------------------------------------------------------

def test_move_away_and_return_reports_max_excursion():
    collected = build_successful_mission()
    # After completion (receipt > RECEIPT_BASE + 5e9), rover leaves and
    # returns to start: endpoint delta is ~0 but max excursion is not.
    post = [
        make_odom(
            RECEIPT_BASE + 6_000_000_000, 140.0, 1.0, 0.0, 0.0,
            frame='base_footprint'),
        make_odom(
            RECEIPT_BASE + 8_000_000_000, 145.0, 0.0, 0.0, 0.0,
            frame='base_footprint'),
    ]
    collected['/ground_truth/odometry'] = post
    result, passed, reasons = me.evaluate_mission(collected)
    assert abs(result['ground_truth_motion_after_completion_m'] - 1.0) < 1e-6
    assert not passed
    assert any('0.01' in r for r in reasons)


# ---------------------------------------------------------------------------
# V16.3 evaluation policy: warnings vs hard failures
# ---------------------------------------------------------------------------

def _tf_seq(steps, base_t_ns=RECEIPT_BASE + 3_000_000_000,
            sim_start=80.0, dt_sim=5.0, base_x=0.01, base_y=0.0, yaw=0.0):
    """
    Build map->odom transforms with the given per-step deltas.

    `steps` is a list of (dx, dy, dyaw_deg) applied successively. The initial
    base pose is recorded first, then each step is applied and the resulting
    pose recorded, so N steps yield N+1 transforms and N real correction
    steps (the largest one a true step between two recorded transforms).
    """
    from geometry_msgs.msg import Quaternion, TransformStamped
    from geometry_msgs.msg import Vector3
    from tf2_msgs.msg import TFMessage
    transforms = []
    yaw = float(yaw)

    def _append(x, y, yaw_deg, t, t_ns):
        qz = math.sin(math.radians(yaw_deg) / 2.0)
        qw = math.cos(math.radians(yaw_deg) / 2.0)
        tr = TransformStamped()
        tr.header.stamp.sec = int(t)
        tr.header.stamp.nanosec = int((t - int(t)) * 1e9)
        tr.header.frame_id = 'map'
        tr.child_frame_id = 'odom'
        tr.transform.translation = Vector3(x=x, y=y, z=0.0)
        tr.transform.rotation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        msg = TFMessage()
        msg.transforms.append(tr)
        transforms.append((t_ns, msg))

    x, y = base_x, base_y
    t, t_ns = sim_start, base_t_ns
    _append(x, y, math.degrees(yaw), t, t_ns)
    for dx, dy, dyaw_deg in steps:
        x += dx
        y += dy
        yaw += math.radians(dyaw_deg)
        t += dt_sim
        t_ns += 1_000_000_000
        _append(x, y, math.degrees(yaw), t, t_ns)
    return transforms


def test_evaluation_policy_version_is_two():
    collected = build_successful_mission()
    result, _, _ = me.evaluate_mission(collected)
    assert result['evaluation_policy_version'] == 2


def test_warnings_and_failure_reasons_are_separate_lists():
    collected = build_successful_mission()
    result, _, reasons = me.evaluate_mission(collected)
    assert isinstance(result['warnings'], list)
    assert isinstance(result['failure_reasons'], list)
    assert result['warnings'] is not result['failure_reasons']
    assert reasons == result['failure_reasons']


def test_coverage_below_98_passes_with_warning():
    collected = build_successful_mission()
    collected['/map'] = [
        (RECEIPT_BASE + 2_000_000_000, make_map(97.93))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert passed, reasons
    assert not reasons
    assert any('known map' in w and '< 98' in w
               for w in result['warnings'])
    assert result['known_map_percent'] < 98.0


def test_coverage_above_98_no_warning():
    collected = build_successful_mission()
    result, passed, reasons = me.evaluate_mission(collected)
    assert passed
    coverage_warnings = [w for w in result['warnings']
                         if 'known map' in w]
    assert not coverage_warnings


def test_permanent_region_without_selectable_frontier_passes():
    collected = build_successful_mission()
    result = __import__('json').dumps(
        result_payload(completion_time_s=75.0, permanent_failed_regions=1)
    )
    collected['/exploration_result'] = [
        (
            RECEIPT_BASE + 5_000_000_001,
            make_string(result),
        )
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert result['permanent_failed_regions'] == 1
    assert result['selectable_frontier_components'] == 0
    assert passed, reasons
    assert any('permanent_failed_regions' in w
               for w in result['warnings'])


def test_five_cell_frontier_component_still_fails():
    from nav_msgs.msg import OccupancyGrid
    collected = build_successful_mission()
    m = OccupancyGrid()
    size = 9
    m.info.width = size
    m.info.height = size
    data = [-1] * (size * size)
    for i in range(5):
        data[i] = 0
    m.data = data
    collected['/map'] = [(RECEIPT_BASE + 2_000_000_000, m)]
    result, passed, reasons = me.evaluate_mission(collected)
    assert any(s >= 5 for s in result['final_frontier_component_sizes'])
    assert not passed
    assert any('>= 5' in r for r in reasons)


def test_max_translation_0_1638_p99_0_0418_yaw_0_4_passes_with_warning():
    collected = build_successful_mission()
    # ~100 small steps of 0.04 m / 0.0 deg, then one outlier 0.1638 m / 0.4 deg.
    # A large sample keeps the p99 of the small routine steps below 0.05 m
    # while the lone outlier is the maximum translation step.
    steps = [(0.04, 0.0, 0.0)] * 100 + [(0.1638, 0.0, 0.4)]
    collected['/tf'] = _tf_seq(steps)
    result, passed, reasons = me.evaluate_mission(collected)
    assert abs(result['maximum_map_to_odom_translation_step_m'] - 0.1638) < 1e-6
    assert abs(result['maximum_map_to_odom_yaw_step_deg'] - 0.4) < 1e-6
    assert result['p99_map_to_odom_translation_step_m'] < 0.05
    assert passed, reasons
    assert any('> 0.15' in w for w in result['warnings'])
    assert not reasons


def test_max_translation_above_0_25_fails():
    collected = build_successful_mission()
    steps = [(0.3, 0.0, 0.0), (0.01, 0.0, 0.0)]
    collected['/tf'] = _tf_seq(steps)
    result, passed, reasons = me.evaluate_mission(collected)
    assert abs(result['maximum_map_to_odom_translation_step_m'] - 0.3) < 1e-6
    assert not passed
    assert any('0.25' in r for r in reasons)


def test_translation_p99_above_0_05_fails():
    collected = build_successful_mission()
    steps = [(0.10, 0.0, 0.0)] * 20
    collected['/tf'] = _tf_seq(steps)
    result, passed, reasons = me.evaluate_mission(collected)
    assert result['p99_map_to_odom_translation_step_m'] > 0.05
    assert result['maximum_map_to_odom_translation_step_m'] <= 0.25
    assert not passed
    assert any('p99' in r for r in reasons)


def test_max_yaw_step_above_5_fails():
    collected = build_successful_mission()
    steps = [(0.01, 0.0, 6.0)]
    collected['/tf'] = _tf_seq(steps)
    result, passed, reasons = me.evaluate_mission(collected)
    assert abs(result['maximum_map_to_odom_yaw_step_deg'] - 6.0) < 1e-6
    assert not passed
    assert any('5.0' in r for r in reasons)


def test_historical_map_splitting_fails():
    collected = build_successful_mission()
    steps = [(0.72, 0.0, 14.4)]
    collected['/tf'] = _tf_seq(steps)
    result, passed, reasons = me.evaluate_mission(collected)
    assert not passed
    assert any('0.25' in r for r in reasons)
    assert any('5.0' in r for r in reasons)


def test_percentile_helper_covers_edge_cases():
    assert me.compute_percentile([], 0.99) == 0.0
    assert me.compute_percentile([5.0], 0.5) == 5.0
    assert me.compute_percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert me.compute_percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert abs(me.compute_percentile([10.0, 20.0], 0.99) - 19.9) < 1e-9
    assert me.compute_percentile([7.0, 7.0, 7.0], 0.99) == 7.0


def test_final_component_size_five_fails_boundary():
    # A single final component of exactly 5 cells is a hard failure
    # (unresolved region). Four cells is NOT.
    result_5 = _build_evaluator_with_components([5])
    assert result_5['passed'] is False
    assert any('>= 5 cells' in r for r in result_5['failure_reasons'])

    result_4 = _build_evaluator_with_components([4])
    assert result_4['passed'] is True
    assert not any('>= 5 cells' in r for r in result_4['failure_reasons'])


def test_permanent_warning_reports_factual_counts():
    # The permanent-region warning must NOT claim geometric frontier is
    # zero, and must NOT use the obsolete 'no selectable final frontier
    # remains' text. It reports the real remaining frontier counts.
    result = _build_evaluator_with_components(
        [10, 3], perm_regions=1
    )
    assert result['passed'] is False  # 10-cell component still fails
    perm_warns = [
        w for w in result['warnings']
        if 'permanent_failed_regions' in w
    ]
    assert perm_warns, 'expected a permanent-region warning'
    text = perm_warns[0]
    assert 'no selectable final frontier remains' not in text
    assert 'geometric frontier still' in text
    assert '13 cells' in text
    assert '1 geometrically selectable' in text


def _build_evaluator_with_components(sizes, perm_regions=0):
    """
    Evaluate a synthetic mission with pinned frontier component sizes.

    The final map reports exactly the requested disconnected frontier
    component sizes, with the requested permanent-region count. The base
    scenario is otherwise clean (valid completion transition, no
    motion/odometry faults), so the only gates exercised are the
    independent geometric final-frontier-component hard check and the
    permanent-region warning text. Component sizes are pinned directly
    (the map clustering is an implementation detail irrelevant to the
    pass/fail policy being tested).
    """
    from unittest.mock import patch

    collected = build_successful_mission()

    # Pin the final-frontier component accounting deterministically.
    total_cells = sum(sizes)
    with patch.object(
        me, 'count_frontier_components',
        lambda data, w, h: (total_cells, list(sizes)),
    ):
        # When permanent regions are requested, embed the count in the
        # /exploration_result JSON the evaluator reads.
        if perm_regions:
            payload = result_payload(
                completion_time_s=75.0,
                permanent_failed_regions=perm_regions,
            )
            collected['/exploration_result'] = [
                (
                    RECEIPT_BASE + 5_000_000_001,
                    make_string(__import__('json').dumps(payload)),
                ),
            ]
        result, passed, reasons = me.evaluate_mission(collected)
    result['passed'] = passed
    result['failure_reasons'] = reasons
    return result


def test_coverage_warning_uses_single_constant():
    # COVERAGE_WARN_PERCENT is the single source of truth for the diagnostic
    # coverage warning; MIN_KNOWN_MAP_PERCENT must no longer exist.
    assert hasattr(me, 'COVERAGE_WARN_PERCENT')
    assert not hasattr(me, 'MIN_KNOWN_MAP_PERCENT')
    # Just below the threshold -> warning only (still passes).
    collected = build_successful_mission()
    collected['/map'] = [
        (RECEIPT_BASE + 2_000_000_000, make_map(97.93))
    ]
    result, passed, reasons = me.evaluate_mission(collected)
    assert passed, reasons
    assert not reasons
    assert any(
        'known map' in w and f'< {me.COVERAGE_WARN_PERCENT}' in w
        for w in result['warnings']
    )
    # At/above the threshold -> no coverage warning.
    collected2 = build_successful_mission()
    r2, passed2, _ = me.evaluate_mission(collected2)
    assert passed2
    assert not any('known map' in w for w in r2['warnings'])


# ---------------------------------------------------------------------------
# Yaw unwrapping across multiple rotations (>720 deg)
# ---------------------------------------------------------------------------

def test_yaw_unwrap_beyond_720_degrees():
    # Monotonic steps of 1.0 rad avoid the +/-pi ambiguity; 14 points
    # span 13.0 rad (~745 deg) > 720 deg and must stay continuous.
    angles = [float(k) for k in range(14)]
    out = me.unwrap_sequence(angles)
    assert out[-1] == 13.0
    assert out[-1] > 4.0 * math.pi


def test_alignment_handles_multirotation_yaw():
    # Filtered and ground truth both rotate >720 deg; yaw error ~ 0.
    # Consecutive samples differ by < pi so unwrapping is unambiguous.
    filtered, gt = [], []
    for k in range(13):
        s = k * 1.0
        yaw = k * (4 * math.pi / 12.0)  # up to 4pi over the sequence
        filtered.append((s, math.cos(yaw), math.sin(yaw), yaw))
        gt.append((s, math.cos(yaw), math.sin(yaw), yaw))
    max_yaw, max_pos = me.align_trajectories(filtered, gt)
    assert max_yaw < 1e-6
    assert max_pos < 1e-6


# ---------------------------------------------------------------------------
# Cumulative relative-yaw error (no re-wrap into +/-180 deg)
# ---------------------------------------------------------------------------

def test_cumulative_yaw_error_both_rotate_720():
    # Both complete >720 deg of rotation; reported error ~ 0 (not wrapped).
    filtered, gt = [], []
    for k in range(13):
        s = k * 1.0
        yaw = k * (4 * math.pi / 12.0)
        filtered.append((s, 0.0, 0.0, yaw))
        gt.append((s, 0.0, 0.0, yaw))
    max_yaw, _ = me.align_trajectories(filtered, gt)
    assert max_yaw < 1e-6


def test_cumulative_yaw_error_relative_720():
    # Filtered rotates 720 deg relative to STATIONARY ground truth.
    # Reported maximum error must be ~720 deg, not clamped to 180 deg.
    filtered, gt = [], []
    for k in range(13):
        s = k * 1.0
        yaw = k * (4 * math.pi / 12.0)
        filtered.append((s, 0.0, 0.0, yaw))
        gt.append((s, 0.0, 0.0, 0.0))  # stationary
    max_yaw, _ = me.align_trajectories(filtered, gt)
    assert abs(max_yaw - 4 * math.pi) < 1e-6


# ---------------------------------------------------------------------------
# Rigid trajectory-frame alignment (odom vs Gazebo world frames)
# ---------------------------------------------------------------------------

def _traj(initial, final, n=11):
    """Linear motion from initial (x, y, yaw) to final (x, y, yaw)."""
    pts = []
    for k in range(n):
        f = k / (n - 1)
        x = initial[0] + f * (final[0] - initial[0])
        y = initial[1] + f * (final[1] - initial[1])
        yaw = initial[2] + f * (final[2] - initial[2])
        pts.append((float(k), x, y, yaw))
    return pts


def test_alignment_cardinal_frame_offset():
    # Filtered odom and ground truth represent the SAME one-metre forward
    # motion but in frames offset by (5, 2) and 90 deg. Errors ~ 0.
    filtered = _traj((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    gt = _traj((5.0, 2.0, math.pi / 2), (5.0, 3.0, math.pi / 2))
    max_yaw, max_pos = me.align_trajectories(filtered, gt)
    assert max_pos < 1e-6
    assert max_yaw < 1e-6


def test_alignment_arbitrary_frame_offset():
    # Same motion, arbitrary non-cardinal frame offset (37 deg).
    offset = math.radians(37.0)
    filtered = _traj((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    # Ground truth shares the initial 37 deg yaw offset and a translation.
    gt = _traj(
        (4.0, -1.0, offset),
        (4.0 + math.cos(offset), -1.0 + math.sin(offset), offset),
    )
    max_yaw, max_pos = me.align_trajectories(filtered, gt)
    assert max_pos < 1e-6
    assert max_yaw < 1e-6


def test_alignment_is_not_quadratic():
    # Two-pointer + binary search: O(n log m), not nested full scan.
    import time
    filtered = [(float(k), float(k), 0.0, 0.0) for k in range(2000)]
    gt = [(float(k) + 0.5, float(k), 0.0, 0.0) for k in range(2000)]
    t0 = time.perf_counter()
    me.align_trajectories(filtered, gt)
    elapsed = time.perf_counter() - t0
    # Should finish in well under a second (linear-ish, not n^2).
    assert elapsed < 1.0
