"""
Evaluate a recorded rosbag2 mission and emit a JSON verdict.

Reads a rosbag2 (MCAP storage) recorded during an exploration mission and
produces a single machine-readable JSON verdict. It uses only native ROS
APIs (rosbag2_py, rclpy.serialization, rosidl_runtime_py) -- no third-party
mcap decoder.

Bag reading is intentionally separated from the pure metric and pass/fail
logic so the latter can be unit tested with synthetic messages.
"""

import argparse
import json
import math
import os
import sys

from rclpy.serialization import deserialize_message
from rosbag2_py import (
    ConverterOptions,
    SequentialReader,
    StorageOptions,
)
import rosidl_runtime_py.utilities

# Planner definitions are reused verbatim so the evaluator's frontier
# accounting can never silently diverge from the runtime node.
from rover_exploration.frontier_detection import (
    cluster_frontier_cells,
    find_frontier_cells,
)

REQUIRED_TOPICS = [
    '/clock',
    '/exploration_complete',
    '/exploration_result',
    '/map',
    '/planned_path',
    '/cmd_vel',
    '/cmd_vel_raw',
    '/odometry/filtered',
    '/ground_truth/odometry',
    '/tf',
    '/recovery_request',
]

# Result schema: exact keys and their expected JSON types.
RESULT_KEYS = [
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
]

# Threshold constants.
#
# Hard mission-failure gates (unchanged from the original evaluator).
MIN_CLUSTER_SIZE = 5
MAX_POST_COMPLETION_OBSERVATION_S = 2.0
MAX_POST_COMPLETION_DISPLACEMENT_M = 0.01
MAX_FILTERED_YAW_ERROR_DEG = 1.0
MAX_MAP_TO_ODOM_YAW_STEP_DEG = 5.0
ACTIVE_CMD_VEL_THRESHOLD = 0.001

# V16.3 evaluation policy: separate diagnostic warnings from hard failures.
EVALUATION_POLICY_VERSION = 2

# Raw rectangular-grid coverage is NOT spawn-invariant (it includes SLAM
# bounding-grid padding, inaccessible obstacle interiors, and map-expansion
# borders). Below this it is a diagnostic warning only, never a failure.
COVERAGE_WARN_PERCENT = 98.0

# A permanent blacklist region is intended behavior for repeatedly invalid or
# unreachable goals; it is a diagnostic warning, not a failure. A genuinely
# unresolved region is independently caught by the raw final-frontier-component
# hard gate (MIN_CLUSTER_SIZE).

# map -> odom translation correction policy (V16.3):
#   * maximum step above 0.15 m  -> diagnostic warning (original quality target)
#   * maximum step above 0.25 m  -> hard failure (severe single correction)
#   * p99 step above 0.05 m      -> hard failure (large corrections routine)
TRANSLATION_STEP_WARN_M = 0.15
TRANSLATION_STEP_FAIL_M = 0.25
TRANSLATION_P99_FAIL_M = 0.05


def compute_percentile(values, q):
    """
    Dependency-free linear percentile using index (n - 1) * q.

    `q` is in [0.0, 1.0]. Returns 0.0 for empty input. Interpolates linearly
    between the two surrounding sorted samples, matching numpy-style quantile
    with the "linear" (type-7) convention for (n - 1) * q spacing.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return float(ordered[0])
    rank = (n - 1) * float(q)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


def yaw_from_quaternion(x, y, z, w):
    """Extract yaw (z-rotation) from a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def unwrap(angle):
    """Unwrap a single angle into (-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def unwrap_sequence(angles):
    """Unwrap a sequence of angles so consecutive deltas stay small."""
    out = []
    prev_raw = None
    prev_out = None
    for angle in angles:
        if angle is None:
            out.append(None)
            continue
        if prev_raw is None:
            out.append(angle)
            prev_raw = angle
            prev_out = angle
            continue
        delta = angle - prev_raw
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta <= -math.pi:
            delta += 2.0 * math.pi
        prev_out = prev_out + delta
        out.append(prev_out)
        prev_raw = angle
    return out


def is_active_cmd_vel(cmd):
    """Return True if the command's linear/angular speed exceeds 0.001."""
    if cmd is None:
        return False
    return math.hypot(cmd.linear.x, cmd.angular.z) > ACTIVE_CMD_VEL_THRESHOLD


def dedup_tf_by_stamp(transforms):
    """
    Deduplicate map->odom TF records by transform timestamp.

    Returns a list of (stamp_s, tx, ty, yaw) sorted by stamp, keeping the
    last record seen for each identical transform timestamp.
    """
    by_stamp = {}
    for stamp_s, tx, ty, yaw in transforms:
        by_stamp[stamp_s] = (stamp_s, tx, ty, yaw)
    return [by_stamp[k] for k in sorted(by_stamp)]


def count_frontier_components(data, width, height, min_cluster_size=1):
    """
    Return (cells, all_component_sizes) using the planner's definitions.

    Every connected component is reported (min_cluster_size=1) so residual
    components such as [2, 1, 1] are visible. The caller derives the
    selectable count (components with size >= MIN_CLUSTER_SIZE).
    """
    cells = find_frontier_cells(
        data=data, width=width, height=height
    )
    clusters = cluster_frontier_cells(cells, min_cluster_size=min_cluster_size)
    sizes = sorted((len(cluster) for cluster in clusters), reverse=True)
    return len(cells), sizes


def parse_odom_triplets(msgs):
    """
    Convert (receipt_ns, Odometry) pairs into sorted (sim_s, x, y, yaw).

    The odometry header.stamp is simulation time, so it is used directly as
    the alignment timestamp (no receipt-time mapping required).
    """
    out = []
    for _, msg in msgs:
        pose = msg.pose.pose
        stamp = msg.header.stamp
        sim_s = stamp.sec + stamp.nanosec / 1e9
        out.append(
            (
                sim_s,
                pose.position.x,
                pose.position.y,
                yaw_from_quaternion(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ),
            )
        )
    return sorted(out, key=lambda row: row[0])


def build_receipt_to_sim(clock_msgs):
    """
    Build a sorted [(receipt_ns, sim_s), ...] table from /clock messages.

    Each /clock message records (receipt timestamp, simulation seconds), so
    this table maps any bag receipt time to simulation time.
    """
    pairs = []
    for receipt_ns, msg in clock_msgs:
        sim_s = msg.clock.sec + msg.clock.nanosec / 1e9
        pairs.append((receipt_ns, sim_s))
    pairs.sort(key=lambda row: row[0])
    return pairs


def receipt_to_sim(receipt_ns, pairs):
    """
    Map a bag receipt timestamp (ns) to simulation seconds.

    Uses sorted linear interpolation between the surrounding /clock samples.
    Receipt times outside the table clamp to the nearest endpoint.
    """
    if not pairs:
        return 0.0
    if receipt_ns <= pairs[0][0]:
        return pairs[0][1]
    if receipt_ns >= pairs[-1][0]:
        return pairs[-1][1]

    lo, hi = 0, len(pairs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pairs[mid][0] <= receipt_ns:
            lo = mid
        else:
            hi = mid
    n0, s0 = pairs[lo]
    n1, s1 = pairs[hi]
    if n1 == n0:
        return s0
    frac = (receipt_ns - n0) / (n1 - n0)
    return s0 + frac * (s1 - s0)


def _gt_bracket(timestamps, t):
    """Return (lo_idx, hi_idx) bracketing t in a sorted timestamp list."""
    lo, hi = 0, len(timestamps) - 1
    if t <= timestamps[0]:
        return 0, 0
    if t >= timestamps[-1]:
        return hi, hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if timestamps[mid] <= t:
            lo = mid
        else:
            hi = mid
    return lo, hi


def align_trajectories(filtered, ground_truth):
    """
    Align two (sim_s, x, y, yaw) trajectories and return per-sample deltas.

    Both sequences are first fully unwrapped so yaw stays continuous across
    multiple rotations (>720 deg). For each filtered sample the matching
    ground-truth sample is found by binary search (O(log n) per lookup),
    with linear interpolation between bracketing ground-truth samples.

    /odometry/filtered is expressed in the EKF's local `odom` frame while
    /ground_truth/odometry is expressed in Gazebo's `world` frame; these
    frames can have different initial positions and orientations. After
    interpolating ground truth at each filtered timestamp, the ground-truth
    relative displacement is rotated by the initial yaw difference so it is
    compared in the filtered odometry frame. Once both sequences are unwrapped
    the relative-yaw difference is NOT re-wrapped. Returns
    (max_yaw_error_rad, max_position_error_m).
    """
    if not filtered or not ground_truth:
        return 0.0, 0.0

    f_yaws = unwrap_sequence([yaw for _, _, _, yaw in filtered])
    g_yaws = unwrap_sequence([yaw for _, _, _, yaw in ground_truth])
    f_pts = [
        (t, x, y, f_yaws[i]) for i, (t, x, y, _) in enumerate(filtered)
    ]
    g_pts = [
        (t, x, y, g_yaws[i]) for i, (t, x, y, _) in enumerate(ground_truth)
    ]

    f0_t, f0_x, f0_y, f0_yaw = f_pts[0]
    g0_t, g0_x, g0_y, g0_yaw = g_pts[0]
    g_times = [t for t, _, _, _ in g_pts]

    # Rotate ground-truth relative displacement into the filtered frame.
    alignment_yaw = f0_yaw - g0_yaw
    c_align, s_align = math.cos(alignment_yaw), math.sin(alignment_yaw)

    max_yaw_error = 0.0
    max_position_error = 0.0

    for t_f, fx, fy, fyaw in f_pts:
        lo, hi = _gt_bracket(g_times, t_f)
        t0, gx0, gy0, gyaw0 = g_pts[lo]
        if lo == hi:
            gx, gy, gyaw = gx0, gy0, gyaw0
        else:
            t1, gx1, gy1, gyaw1 = g_pts[hi]
            if t1 == t0:
                gx, gy, gyaw = gx0, gy0, gyaw0
            else:
                frac = (t_f - t0) / (t1 - t0)
                gx = gx0 + frac * (gx1 - gx0)
                gy = gy0 + frac * (gy1 - gy0)
                gyaw = gyaw0 + frac * (gyaw1 - gyaw0)

        filtered_dx = fx - f0_x
        filtered_dy = fy - f0_y

        gt_dx = gx - g0_x
        gt_dy = gy - g0_y
        aligned_gt_dx = c_align * gt_dx - s_align * gt_dy
        aligned_gt_dy = s_align * gt_dx + c_align * gt_dy

        position_error = math.hypot(
            filtered_dx - aligned_gt_dx, filtered_dy - aligned_gt_dy
        )
        # Both sequences are already unwrapped; do not re-wrap.
        yaw_error = abs(
            (fyaw - f0_yaw) - (gyaw - g0_yaw)
        )

        if position_error > max_position_error:
            max_position_error = position_error
        if yaw_error > max_yaw_error:
            max_yaw_error = yaw_error

    return max_yaw_error, max_position_error


def read_bag(bag_dir):
    """
    Read only the required evaluator topics from a rosbag2 directory.

    Returns {topic: [(receipt_ns, message), ...]}. Raises FileNotFoundError
    / ValueError with a clear message on missing or unreadable bags.
    """
    if not os.path.isdir(bag_dir):
        raise FileNotFoundError(f'Bag directory not found: {bag_dir}')

    storage_options = StorageOptions(uri=bag_dir, storage_id='mcap')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )

    reader = SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as error:  # noqa: BLE001 - surface a clear message
        raise ValueError(
            f'Could not open bag {bag_dir}: {error}'
        ) from error

    topic_types = {
        topic.name: topic.type
        for topic in reader.get_all_topics_and_types()
    }

    missing = [t for t in REQUIRED_TOPICS if t not in topic_types]
    if missing:
        raise ValueError(
            'Bag is missing required topic(s): ' + ', '.join(missing)
        )

    # Only deserialize the topics the evaluator actually needs.
    type_map = {
        name: rosidl_runtime_py.utilities.get_message(typ)
        for name, typ in topic_types.items()
        if name in REQUIRED_TOPICS
    }

    collected = {topic: [] for topic in REQUIRED_TOPICS}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        msg_type = type_map.get(topic)
        if msg_type is None:
            continue
        message = deserialize_message(data, msg_type)
        collected[topic].append((t_ns, message))

    return collected


def validate_result(parsed):
    """
    Validate a parsed /exploration_result object against the schema.

    Returns (ok, reason). Rejects empty/missing/extra keys, wrong types,
    wrong schema version, completed != True, non-finite/negative
    completion_time_s, or non-integer/negative counters.
    """
    if not isinstance(parsed, dict):
        return False, 'result is not a JSON object'
    if set(parsed.keys()) != set(RESULT_KEYS):
        return False, 'result keys do not match the required schema'
    if parsed.get('schema_version') != 1:
        return False, 'result schema_version != 1'
    if parsed.get('completed') is not True:
        return False, 'result completed != true'

    # completion_time_s: numeric, finite, non-negative, not a bool.
    completion_time = parsed.get('completion_time_s')
    if isinstance(completion_time, bool) or not isinstance(
        completion_time, (int, float)
    ):
        return False, 'completion_time_s is not numeric'
    if not math.isfinite(completion_time) or completion_time < 0.0:
        return False, 'completion_time_s is not a finite non-negative number'

    # Counters: integers (not bools), non-negative.
    for key in (
        'goals_assigned',
        'goals_reached',
        'failure_events',
        'temporary_failure_events',
        'permanent_failed_regions',
        'recovery_requests',
        'visited_regions',
        'frontier_cells',
        'frontier_clusters',
    ):
        value = parsed.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return False, f'{key} is not an integer'
        if value < 0:
            return False, f'{key} is negative'

    return True, ''


def evaluate_mission(collected):
    """
    Compute the mission verdict from collected bag messages.

    Returns (result_dict, passed, failure_reasons).
    """
    failure_reasons = []
    warnings = []

    completion_msgs = sorted(
        collected.get('/exploration_complete', []), key=lambda r: r[0]
    )
    clock_pairs = build_receipt_to_sim(collected.get('/clock', []))

    # ---- Completion state machine validation ----
    values = [bool(m.data) for _, m in completion_msgs]
    # All recorded states must be present; the FIRST recorded state must be
    # false, exactly one false->true transition must occur, and the final
    # recorded state must be true.
    initial_state_is_false = bool(values) and values[0] is False
    transitions = 0
    prev = None
    completion_receipt_ns = None
    parsed_result = None
    result_outcome = 'success'
    for t_ns, msg in completion_msgs:
        value = bool(msg.data)
        if prev is not None and (not prev) and value:
            transitions += 1
            if completion_receipt_ns is None:
                completion_receipt_ns = t_ns
        prev = value
    final_complete = bool(values[-1]) if values else False

    # ---- Result association ----
    # Sort result messages by receipt time, then pick the first valid
    # result whose completion_time_s agrees with the mapped completion
    # transition time within a documented tolerance.
    parsed_result = None
    if completion_receipt_ns is not None:
        result_candidates = [
            (t_ns, msg)
            for t_ns, msg in sorted(
                collected.get('/exploration_result', []), key=lambda r: r[0]
            )
            if t_ns >= completion_receipt_ns and msg.data
        ]
        completion_time_s = receipt_to_sim(
            completion_receipt_ns, clock_pairs
        )
        for t_ns, msg in result_candidates:
            try:
                candidate = json.loads(msg.data)
            except (ValueError, TypeError):
                continue
            ok, _ = validate_result(candidate)
            if not ok:
                continue
            if abs(candidate.get('completion_time_s', float('nan'))
                   - completion_time_s) <= 0.5:
                parsed_result = candidate
                result_outcome = candidate.get('outcome', 'success')
                break

    # ---- Map / frontier accounting (final map) ----
    map_msgs = sorted(collected.get('/map', []), key=lambda r: r[0])
    known_map_percent = 0.0
    final_frontier_cells = 0
    final_component_sizes = []
    if map_msgs:
        final_map = map_msgs[-1][1]
        width = final_map.info.width
        height = final_map.info.height
        total = width * height
        if total:
            known = sum(1 for v in final_map.data if v != -1)
            known_map_percent = 100.0 * known / total
        final_frontier_cells, final_component_sizes = (
            count_frontier_components(final_map.data, width, height)
        )

    selectable_components = sum(
        1 for size in final_component_sizes if size >= MIN_CLUSTER_SIZE
    )

    # ---- Completion simulation time and post-observation duration ----
    completion_time_s = receipt_to_sim(
        completion_receipt_ns, clock_pairs
    ) if completion_receipt_ns is not None else 0.0

    if clock_pairs:
        final_clock_sim_s = clock_pairs[-1][1]
    else:
        final_clock_sim_s = 0.0
    post_completion_s = (
        final_clock_sim_s - completion_time_s
        if completion_receipt_ns is not None
        else 0.0
    )

    # ---- After-completion command / path checks (receipt-time domain) ----
    active_cmd_vel_after = 0
    active_cmd_vel_raw_after = 0
    nonempty_paths_after = 0

    def after_completion(t_ns):
        return (
            completion_receipt_ns is None
            or t_ns > completion_receipt_ns
        )

    for t_ns, msg in collected.get('/cmd_vel', []):
        if after_completion(t_ns) and is_active_cmd_vel(msg):
            active_cmd_vel_after += 1
    for t_ns, msg in collected.get('/cmd_vel_raw', []):
        if after_completion(t_ns) and is_active_cmd_vel(msg):
            active_cmd_vel_raw_after += 1
    for t_ns, msg in collected.get('/planned_path', []):
        if after_completion(t_ns) and len(msg.poses) > 0:
            nonempty_paths_after += 1

    # ---- Ground-truth displacement after completion (max excursion) ----
    gt_raw = collected.get('/ground_truth/odometry', [])
    post_gt = [
        (t_ns, msg)
        for t_ns, msg in gt_raw
        if after_completion(t_ns)
    ]
    gt_motion_after = 0.0
    if len(post_gt) >= 1:
        first_pose = post_gt[0][1].pose.pose
        x0, y0 = first_pose.position.x, first_pose.position.y
        gt_motion_after = 0.0
        for _, msg in post_gt:
            x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
            d = math.hypot(x - x0, y - y0)
            if d > gt_motion_after:
                gt_motion_after = d

    # ---- Filtered vs ground-truth trajectory errors ----
    filtered = parse_odom_triplets(
        collected.get('/odometry/filtered', [])
    )
    gt_traj = parse_odom_triplets(gt_raw)
    max_yaw_error_rad, max_position_error_m = align_trajectories(
        filtered, gt_traj
    )
    max_yaw_error_deg = max_yaw_error_rad * 180.0 / math.pi

    # ---- map -> odom correction steps ----
    map_to_odom = []
    for topic in ('/tf', '/tf_static'):
        for t_ns, tf_msg in collected.get(topic, []):
            for tr in tf_msg.transforms:
                if (
                    tr.header.frame_id == 'map'
                    and tr.child_frame_id == 'odom'
                ):
                    ts = (
                        tr.header.stamp.sec
                        + tr.header.stamp.nanosec / 1e9
                    )
                    map_to_odom.append(
                        (
                            ts,
                            tr.transform.translation.x,
                            tr.transform.translation.y,
                            yaw_from_quaternion(
                                tr.transform.rotation.x,
                                tr.transform.rotation.y,
                                tr.transform.rotation.z,
                                tr.transform.rotation.w,
                            ),
                        )
                    )
    map_to_odom = dedup_tf_by_stamp(map_to_odom)

    trans_steps = []
    max_trans_step = 0.0
    max_yaw_step_deg = 0.0
    for i in range(1, len(map_to_odom)):
        _, px, py, pyaw = map_to_odom[i - 1]
        _, cx, cy, cyaw = map_to_odom[i]
        step = math.hypot(cx - px, cy - py)
        trans_steps.append(step)
        max_trans_step = max(max_trans_step, step)
        max_yaw_step_deg = max(
            max_yaw_step_deg,
            abs(unwrap(cyaw - pyaw)) * 180.0 / math.pi,
        )

    # p99 of the per-step translation corrections: measures whether large
    # corrections are routine rather than a single outlier.
    p99_trans_step = compute_percentile(trans_steps, 0.99)

    goals_assigned = 0
    goals_reached = 0
    temp_fail = 0
    perm_regions = 0
    recovery_requests = 0
    if parsed_result:
        goals_assigned = parsed_result.get('goals_assigned', 0)
        goals_reached = parsed_result.get('goals_reached', 0)
        temp_fail = parsed_result.get('temporary_failure_events', 0)
        perm_regions = parsed_result.get('permanent_failed_regions', 0)
        recovery_requests = parsed_result.get('recovery_requests', 0)

    # ---- Telemetry sufficiency ----
    # A required topic existing in metadata does not prove it carries data.
    # Zero/insufficient messages must fail rather than silently pass.
    clock_msgs = collected.get('/clock', [])
    filtered_msgs = collected.get('/odometry/filtered', [])
    gt_msgs_full = collected.get('/ground_truth/odometry', [])
    if len(clock_msgs) < 2:
        failure_reasons.append(
            'fewer than two /clock messages: cannot map receipt to sim time'
        )
    if not map_msgs:
        failure_reasons.append('no final /map message recorded')
    if len(filtered_msgs) < 2:
        failure_reasons.append(
            'fewer than two /odometry/filtered messages: '
            'localization error unsubstantiated'
        )
    if len(gt_msgs_full) < 2:
        failure_reasons.append(
            'fewer than two /ground_truth/odometry messages: '
            'ground truth unsubstantiated'
        )
    if len(map_to_odom) < 2:
        failure_reasons.append(
            'fewer than two distinct map->odom transforms: '
            'correction step unsubstantiated'
        )
    if len(post_gt) < 2:
        failure_reasons.append(
            'fewer than two ground-truth samples after completion: '
            'stopping unsubstantiated'
        )

    result = {
        'schema_version': 1,
        'evaluation_policy_version': EVALUATION_POLICY_VERSION,
        'passed': False,
        'failure_reasons': failure_reasons,
        'warnings': warnings,
        'completion_count': transitions,
        'completion_time_s': completion_time_s,
        'post_completion_observation_s': post_completion_s,
        'known_map_percent': known_map_percent,
        'final_frontier_cells': final_frontier_cells,
        'final_frontier_component_sizes': final_component_sizes,
        'selectable_frontier_components': selectable_components,
        'result_outcome': result_outcome,
        'goals_assigned': goals_assigned,
        'goals_reached': goals_reached,
        'temporary_failure_events': temp_fail,
        'permanent_failed_regions': perm_regions,
        'recovery_requests': recovery_requests,
        'active_cmd_vel_after_completion': active_cmd_vel_after,
        'active_cmd_vel_raw_after_completion': active_cmd_vel_raw_after,
        'nonempty_paths_after_completion': nonempty_paths_after,
        'ground_truth_motion_after_completion_m': gt_motion_after,
        'maximum_filtered_yaw_error_deg': max_yaw_error_deg,
        'maximum_filtered_position_error_m': max_position_error_m,
        'maximum_map_to_odom_translation_step_m': max_trans_step,
        'p99_map_to_odom_translation_step_m': p99_trans_step,
        'maximum_map_to_odom_yaw_step_deg': max_yaw_step_deg,
    }

    # ---- Pass criteria (each produces a clear, independent reason) ----
    if not initial_state_is_false:
        failure_reasons.append(
            'first recorded /exploration_complete state is not false'
        )
    if transitions != 1:
        failure_reasons.append(
            f'expected exactly one false->true completion transition, '
            f'got {transitions}'
        )
    if not final_complete:
        failure_reasons.append(
            'final /exploration_complete state is not true'
        )
    if parsed_result is None:
        failure_reasons.append(
            'valid /exploration_result (completed=true) not found'
        )
    if any(size >= MIN_CLUSTER_SIZE for size in final_component_sizes):
        failure_reasons.append(
            'a final frontier component contains '
            f'>= {MIN_CLUSTER_SIZE} cells'
        )
    # Raw rectangular-grid coverage is diagnostic only: it is not
    # spawn-invariant (SLAM bounding-grid padding, inaccessible interiors, and
    # map-expansion borders all count as unknown). It never fails the mission.
    if known_map_percent < COVERAGE_WARN_PERCENT:
        warnings.append(
            f'raw rectangular-grid known map {known_map_percent:.2f}% < '
            f'{COVERAGE_WARN_PERCENT:.1f}%; diagnostic only'
        )
    # A permanent blacklist region is intended behavior for repeatedly invalid
    # or unreachable goals; it is a diagnostic warning, not a failure. A
    # genuinely unresolved region is already caught above by the raw
    # final-frontier-component hard gate. Report facts: do NOT claim geometric
    # frontier is zero (it is not), and do NOT call everything "selectable".
    if perm_regions != 0:
        warnings.append(
            f'permanent_failed_regions = {perm_regions}; '
            f'{perm_regions} goal region(s) blacklisted after repeated '
            f'failures (geometric frontier still '
            f'{final_frontier_cells} cells in '
            f'{len(final_component_sizes)} component(s); '
            f'{selectable_components} geometrically selectable by size >= '
            f'{MIN_CLUSTER_SIZE})'
        )
    if post_completion_s < MAX_POST_COMPLETION_OBSERVATION_S:
        failure_reasons.append(
            f'post-completion observation {post_completion_s:.2f}s < '
            f'{MAX_POST_COMPLETION_OBSERVATION_S}s'
        )
    if active_cmd_vel_after > 0:
        failure_reasons.append(
            f'{active_cmd_vel_after} nonzero /cmd_vel after completion'
        )
    if active_cmd_vel_raw_after > 0:
        failure_reasons.append(
            f'{active_cmd_vel_raw_after} nonzero /cmd_vel_raw after '
            'completion'
        )
    if nonempty_paths_after > 0:
        failure_reasons.append(
            f'{nonempty_paths_after} nonempty planned path after completion'
        )
    if gt_motion_after > MAX_POST_COMPLETION_DISPLACEMENT_M:
        failure_reasons.append(
            f'ground-truth displacement after completion '
            f'{gt_motion_after:.4f}m > '
            f'{MAX_POST_COMPLETION_DISPLACEMENT_M}m'
        )
    if max_yaw_error_deg > MAX_FILTERED_YAW_ERROR_DEG:
        failure_reasons.append(
            f'maximum filtered yaw error {max_yaw_error_deg:.3f}deg > '
            f'{MAX_FILTERED_YAW_ERROR_DEG}deg'
        )
    # map -> odom translation corrections: the maximum detects a severe single
    # correction; the p99 measures whether large corrections are routine.
    if max_trans_step > TRANSLATION_STEP_FAIL_M:
        failure_reasons.append(
            f'maximum map->odom translation step {max_trans_step:.4f}m > '
            f'{TRANSLATION_STEP_FAIL_M}m'
        )
    elif max_trans_step > TRANSLATION_STEP_WARN_M:
        warnings.append(
            f'maximum map->odom translation step {max_trans_step:.4f}m > '
            f'{TRANSLATION_STEP_WARN_M}m (quality target; diagnostic only)'
        )
    if p99_trans_step > TRANSLATION_P99_FAIL_M:
        failure_reasons.append(
            f'map->odom translation p99 {p99_trans_step:.4f}m > '
            f'{TRANSLATION_P99_FAIL_M}m (large corrections routine)'
        )
    if max_yaw_step_deg > MAX_MAP_TO_ODOM_YAW_STEP_DEG:
        failure_reasons.append(
            f'maximum map->odom yaw step {max_yaw_step_deg:.3f}deg > '
            f'{MAX_MAP_TO_ODOM_YAW_STEP_DEG}deg'
        )

    passed = len(failure_reasons) == 0
    result['passed'] = passed
    result['failure_reasons'] = failure_reasons
    return result, passed, failure_reasons


def main(args=None):
    """
    Evaluate a mission bag and write the JSON verdict.

    Returns 0 if the mission passed, 1 if it failed the pass criteria,
    and 2 on invalid arguments or an unreadable/malformed bag.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate a rover exploration mission bag.'
    )
    parser.add_argument('bag_directory', help='rosbag2 MCAP directory')
    parser.add_argument(
        '--output-json',
        required=True,
        help='path to write the JSON result',
    )
    parsed = parser.parse_args(args)

    try:
        collected = read_bag(parsed.bag_directory)
    except (FileNotFoundError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2

    try:
        result, passed, _ = evaluate_mission(collected)
    except Exception as error:  # noqa: BLE001 - malformed bag/result
        print(f'ERROR: failed to evaluate mission: {error}', file=sys.stderr)
        return 2

    text = json.dumps(result, sort_keys=True, indent=2)
    print(text)
    try:
        with open(parsed.output_json, 'w') as handle:
            handle.write(text + '\n')
    except OSError as error:
        print(
            f'ERROR: could not write output {parsed.output_json}: {error}',
            file=sys.stderr,
        )
        return 2

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
