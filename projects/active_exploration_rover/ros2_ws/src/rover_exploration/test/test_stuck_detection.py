import math

import pytest

from rover_exploration.grid_planning import (
    find_escape_path,
)
from rover_exploration.stuck_detection import (
    alignment_progress_rad,
    distance_to_goal_m,
    heading_error_rad,
    is_stuck,
    quaternion_yaw,
)


FREE = 0
OCCUPIED = 100
UNKNOWN = -1


def test_already_inflation_safe_returns_single_cell_path():
    raw_data = [FREE] * 3

    path = find_escape_path(
        raw_data=raw_data,
        inflated_data=raw_data,
        width=3,
        height=1,
        start=(0, 1),
    )

    assert path == [(0, 1)]


def test_escapes_through_raw_free_but_inflated_blocked_cells():
    # The rover cell and its neighbours are free on the raw map but
    # blocked by inflation; the escape must walk through them to the
    # nearest inflation-safe cell.
    raw_data = [FREE] * 6

    inflated_data = [FREE] * 6
    inflated_data[0] = OCCUPIED
    inflated_data[1] = OCCUPIED
    inflated_data[2] = OCCUPIED

    path = find_escape_path(
        raw_data=raw_data,
        inflated_data=inflated_data,
        width=6,
        height=1,
        start=(0, 2),
    )

    assert path is not None
    assert path[0] == (0, 2)
    assert path[-1] == (0, 3)

    for row, column in path:
        index = row * 6 + column
        assert raw_data[index] == FREE


def test_refuses_to_cross_raw_occupied_wall():
    # The start is inflated-blocked and a real occupied wall
    # separates it from the only inflation-safe cell. No escape may
    # cross the wall.
    raw_data = [
        FREE, OCCUPIED, FREE,
    ]

    inflated_data = [
        OCCUPIED, OCCUPIED, FREE,
    ]

    assert find_escape_path(
        raw_data=raw_data,
        inflated_data=inflated_data,
        width=3,
        height=1,
        start=(0, 0),
    ) is None


def test_refuses_to_cross_unknown_cells():
    raw_data = [
        FREE, UNKNOWN, FREE,
    ]

    inflated_data = [
        OCCUPIED, UNKNOWN, FREE,
    ]

    assert find_escape_path(
        raw_data=raw_data,
        inflated_data=inflated_data,
        width=3,
        height=1,
        start=(0, 0),
    ) is None


def test_start_on_raw_occupied_cell_returns_none():
    raw_data = [OCCUPIED, FREE]
    inflated_data = [OCCUPIED, FREE]

    assert find_escape_path(
        raw_data=raw_data,
        inflated_data=inflated_data,
        width=2,
        height=1,
        start=(0, 0),
    ) is None


def test_none_start_returns_none():
    assert find_escape_path(
        raw_data=[],
        inflated_data=[],
        width=0,
        height=0,
        start=None,
    ) is None


def test_quaternion_yaw_handles_identity_and_flip():
    assert quaternion_yaw(0.0, 0.0, 0.0, 1.0) == 0.0
    assert abs(quaternion_yaw(0.0, 0.0, 1.0, 0.0)) == __import__(
        'math'
    ).pi


def test_progress_toward_goal_is_not_stuck():
    samples = [
        (0.0, 1.0, 1.0, 0.0),
        (2.0, 1.5, 1.0, 0.0),
        (5.0, 2.0, 1.0, 0.0),
    ]

    assert is_stuck(
        progress_samples=samples,
        goal_position=(4.0, 1.0),
        minimum_window_s=5.0,
        progress_threshold_m=0.05,
    ) is False


def test_new_goal_gets_a_fresh_window():
    # Only a short window exists since the goal was assigned.
    samples = [
        (10.0, 1.0, 1.0, 0.0),
        (11.0, 1.0, 1.0, 0.0),
    ]

    assert is_stuck(
        progress_samples=samples,
        goal_position=(5.0, 1.0),
        minimum_window_s=5.0,
        progress_threshold_m=0.05,
    ) is False


def test_rotation_is_not_immediate_stuck_behavior():
    # Distance to the goal barely changed but the rover turned from
    # facing away (error ~pi) to facing the goal (error 0): a
    # productive alignment maneuver, not stuck.
    goal = (0.0, 1.0)

    samples = [
        (0.0, 1.0, 1.0, 0.0),
        (2.5, 1.0, 1.0, math.pi),
        (5.0, 1.0, 1.0, math.pi - 0.02),
    ]

    assert is_stuck(
        progress_samples=samples,
        goal_position=goal,
        minimum_window_s=5.0,
        progress_threshold_m=0.05,
    ) is False


def test_stationary_alternating_yaw_is_eventually_stuck():
    # The old accumulated-turning criterion wrongly exempted this:
    # yaw oscillates 0, +0.5, -0.5, ... with no net alignment and no
    # position progress, so it must be classified as stuck after the
    # full window.
    goal = (5.0, 1.0)

    yaws = [0.0, 0.5, -0.5, 0.5, -0.5, 0.0]

    samples = [
        (float(index), 1.0, 1.0, yaw)
        for index, yaw in enumerate(yaws)
    ]

    assert is_stuck(
        progress_samples=samples,
        goal_position=goal,
        minimum_window_s=5.0,
        progress_threshold_m=0.05,
    ) is True


def test_heading_error_across_pi_boundary():
    position = (0.0, 0.0)
    goal = (-1.0, 0.0)  # target bearing is exactly +pi

    # A yaw just below +pi is almost aligned.
    almost_aligned = heading_error_rad(
        math.nextafter(math.pi, 0.0), position, goal
    )
    assert almost_aligned < 0.01

    # A yaw just above -pi (wrapped) is also almost aligned.
    wrapped = heading_error_rad(
        -math.nextafter(math.pi, 0.0), position, goal
    )
    assert wrapped < 0.01

    # Opposite heading gives the maximum error of pi.
    assert heading_error_rad(0.0, position, goal) == (
        pytest.approx(math.pi)
    )


def test_alignment_progress_handles_wraparound():
    # Rover starts facing away from the goal and ends nearly aligned
    # after turning across the +-pi boundary.
    position = (0.0, 0.0)
    goal = (-1.0, 0.0)

    samples = [
        (0.0, position[0], position[1], 0.0),
        (5.0, position[0], position[1], math.pi - 0.05),
    ]

    progress = alignment_progress_rad(samples, goal)

    assert progress > math.pi / 8.0


def test_productive_alignment_receives_grace_period():
    # No distance progress yet, but the rover turned from facing
    # backwards (error ~pi) to facing the goal (error 0): alignment
    # progress exceeds the threshold so it is not stuck.
    goal = (0.0, 1.0)

    samples = [
        (0.0, 1.0, 1.0, 0.0),
        (5.0, 1.0, 1.0, math.pi / 2.0),
    ]

    assert is_stuck(
        progress_samples=samples,
        goal_position=goal,
        minimum_window_s=5.0,
        progress_threshold_m=0.05,
        alignment_threshold_rad=math.pi / 8.0,
    ) is False


def test_no_progress_goal_blacklisted_only_after_full_window():
    # Full window, no meaningful progress toward the goal and no
    # significant turning: genuinely stuck.
    samples = [
        (0.0, 1.0, 1.0, 0.0),
        (2.5, 1.005, 1.0, 0.01),
        (5.0, 1.01, 1.0, 0.02),
    ]

    assert is_stuck(
        progress_samples=samples,
        goal_position=(5.0, 1.0),
        minimum_window_s=5.0,
        progress_threshold_m=0.05,
    ) is True


def test_distance_to_goal_measures_euclidean_distance():
    assert distance_to_goal_m((0.0, 0.0), (3.0, 4.0)) == 5.0


# Route progress uses a fixed reference within each window, not traveled distance.
def test_detour_progress_away_from_final_goal():
    from rover_exploration.stuck_detection import route_progress
    route=((0.,0.),(-2.,0.),(-2.,-2.),(2.,-2.))
    samples=[(0.,0.,0.,math.pi,1,route),(5.,-.5,0.,math.pi,2,route)]
    assert math.dist(samples[-1][1:3],route[-1])>math.dist(samples[0][1:3],route[-1])
    result=route_progress(samples,4.5,.05)
    assert result['progress_m']==.5 and not result['stuck']


def test_stationary_and_oscillating_routes_remain_bounded():
    from rover_exploration.stuck_detection import route_progress
    route=((0.,0.),(5.,0.))
    for xs in [[0]*7,[0,.3,0,.3,0,.3,0]]:
        samples=[(float(t),x,0.,0.,1,route) for t,x in enumerate(xs)]
        assert route_progress(samples,4.5,.05)['stuck']


def test_replanning_does_not_reset_window_or_create_progress():
    from types import SimpleNamespace
    from rover_exploration.exploration_policy import ExplorationPolicy
    config=SimpleNamespace(stuck_window_s=6.,stuck_progress_threshold_m=.05,
        stuck_alignment_threshold_rad=math.pi/8,blacklist_radius_m=.75,
        blacklist_duration_s=30.,permanent_after_failures=2,permanent_exclusion_radius_m=.2)
    p=ExplorationPolicy(config);p.target=SimpleNamespace(goal_world=(4.,0.))
    for t in range(6):
        # Alternating route direction/length cannot confer alignment or arc credit.
        p.set_route(((0.,0.),((-1)**t*(10-t),0.)))
        event=p.observe_pose(float(t),(0.,0.,0.))
        if t<5:assert event is None
    assert event is not None and p.recovery_state=='requested'
    assert p.progress_measurement['progress_m']==0.
    assert p.progress_measurement['reference_route_version']==1
    assert p.progress_measurement['current_route_version']==6
