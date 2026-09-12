"""Offline regressions for diagnostic references; no ROS nodes or model calls."""
import math
from types import SimpleNamespace
from rover_exploration.exploration_policy import ExplorationPolicy
from rover_exploration.stuck_detection import route_progress


def policy():
    p=ExplorationPolicy(SimpleNamespace(stuck_window_s=6.,stuck_progress_threshold_m=.05,
        stuck_alignment_threshold_rad=math.pi/8,blacklist_radius_m=.75,
        blacklist_duration_s=30.,permanent_after_failures=2,permanent_exclusion_radius_m=.2))
    p.target=SimpleNamespace(goal_world=(2.,-2.))
    return p


def test_detour_then_transient_empty_preserves_reference_not_motion():
    p=policy();route=((0.,0.),(-2.,0.),(-2.,-2.),(2.,-2.))
    p.set_route(route,now_s=0.)
    p.observe_pose(0.,(0.,0.,math.pi))
    p.set_route([],now_s=1.)
    assert p.active_route==() and p.diagnostic_route==route
    p.observe_pose(1.,(-.1,0.,math.pi))
    p.set_route(route,now_s=2.)
    assert p.observe_pose(5.,(-.5,0.,math.pi)) is None
    assert p.progress_measurement['progress_m']==.5
    assert p.progress_measurement['final_goal_distance_progress_m']<0


def test_unavailable_is_not_measured_zero():
    r=route_progress([(0,0,0,0,1,()),(5,0,0,0,2,())],4.5,.05)
    assert r['progress_m'] is None and r['alignment_rad'] is None
    assert not r['stuck'] and r['reason']=='progress_unavailable'


def test_persistent_empty_paths_fail_bounded_even_if_rover_moves():
    p=policy();p.set_route(((0.,0.),(10.,0.)),now_s=0.)
    p.observe_pose(0.,(0.,0.,0.))
    for t in range(1,7):
        p.set_route([],now_s=float(t))
        event=p.observe_pose(float(t),(.2*t,0.,0.))
    assert event and p.progress_measurement['reason']=='planning_route_unavailable'
    assert p.progress_measurement['route_missing_since']==1.


def test_stationary_oscillation_and_replans_do_not_buy_time():
    for xs in [[0]*6,[0,.3,0,.3,0,0]]:
        p=policy()
        for t,x in enumerate(xs):
            p.set_route(((0.,0.),(10.+t,0.)),now_s=float(t))
            event=p.observe_pose(float(t),(x,0.,0.))
        assert event and p.progress_measurement['reason']=='route_no_progress'


def test_goal_change_cannot_reuse_old_reference():
    p=policy();p.set_route(((0.,0.),(10.,0.)),now_s=0.);p.observe_pose(0.,(0.,0.,0.))
    p.target.goal_world=(4.,4.)
    p.set_route([],now_s=4.)
    p.observe_pose(5.,(0.,0.,0.))
    assert not p.diagnostic_route and p.progress_measurement['progress_m'] is None
    assert p.observe_pose(9.,(0.,0.,0.))


def test_cancel_prevents_new_samples_and_reference_reuse():
    p=policy();p.set_route(((0.,0.),(10.,0.)),now_s=0.)
    p.invalidate_progress_reference('cancel',1.,cancel=True)
    p.set_route(((0.,0.),(10.,0.)),now_s=2.)
    assert p.observe_pose(10.,(0.,0.,0.)) is None
    assert not p.progress_samples and not p.diagnostic_route


def test_frame_and_repeated_localization_invalidations_keep_deadline():
    p=policy();p.set_route(((0.,0.),(10.,0.)),now_s=0.);p.observe_pose(0.,(0.,0.,0.))
    p.set_route([],frame='other',now_s=1.)
    assert not p.diagnostic_route and not p.progress_samples
    for t in range(2,7):
        p.invalidate_progress_reference('localization_jump',float(t))
        p.set_route(((0.,0.),(10.,0.)),frame='other',now_s=float(t))
        event=p.observe_pose(float(t),(0.,0.,0.))
    assert event and p.progress_measurement['reason']=='progress_reference_unavailable'
    assert p.progress_measurement['reference_failure_since']==1.


def recorded_replay():
    import json
    from pathlib import Path
    f=json.loads((Path(__file__).parent/'fixtures/empty_reference_63_68.json').read_text())
    p=policy();p.target.goal_world=(8.205036125593624,-.007528186558643846)
    rows=[];old_samples=[];route=();version=0
    for e in f['events']:
        if e['kind']=='route':
            route=tuple(map(tuple,e['route']));version=e['route_version']
            p.set_route(route,frame=e['frame'],now_s=e['path_stamp_s'])
        else:
            old_samples.append((e['sim_s'],*e['pose'],version,route))
            event=p.observe_pose(e['sim_s'],e['pose'])
            rows.append(dict(measurement=p.progress_measurement.copy(),recovery=bool(event)))
    return rows,old_samples


def test_recorded_empty_reference_failure_disappears():
    rows,old_samples=recorded_replay()
    assert old_samples[0][4]==65 and old_samples[0][5]==()
    assert rows[-1]['measurement']['sample_sim_s']==68.032
    assert not any(r['recovery'] for r in rows)
    assert abs(rows[-1]['measurement']['progress_m']-.5658601878639735)<1e-9


def test_empty_publication_does_not_execute_retained_diagnostic_path():
    from rover_exploration.frontier_node import FrontierDetector
    from nav_msgs.msg import OccupancyGrid
    p=policy();messages=[];grid=OccupancyGrid();grid.header.frame_id='map';grid.info.resolution=.05
    n=SimpleNamespace(policy=p,node_time_s=lambda:1.,
        path_publisher=SimpleNamespace(publish=messages.append),
        get_logger=lambda:SimpleNamespace(info=lambda _:None))
    FrontierDetector._publish_path(n,grid.header,grid.info,[(0,0),(0,1)])
    FrontierDetector._publish_path(n,grid.header,grid.info,None)
    assert p.diagnostic_route and not p.active_route
    assert len(messages[0].poses)==2 and not messages[-1].poses


def test_missing_map_pose_returns_empty_plan_without_cancelling_goal():
    p=policy();goal=p.target.goal_world
    p.set_route(((3.1,0.),(3.1,-.5),(8.2,-.5)),now_s=62.954)
    # Recorded TF extrapolation leaves both map pose fields unavailable.
    update=p.update(raw_data=[],planning_data=[],width=0,height=0,resolution=.05,
        origin_x=0.,origin_y=0.,frontier_cells=set(),frontier_clusters=[],
        robot_cell=None,robot_world=None,now_s=63.131)
    assert update.path is None and p.target.goal_world==goal
    p.set_route(update.path or [],now_s=63.131)
    assert not p.active_route and p.diagnostic_route
