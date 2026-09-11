"""Focused deterministic executive checks. Never launches a model or simulation."""
from types import SimpleNamespace
import pytest
from rover_exploration.goal_selector import GoalExecutive, ClassicalSelector
from rover_exploration.frontier_selection import CandidateSet
from hierarchical_driver import PersistentDriver


def fixture():
    events=[];requests=[]
    executive=GoalExecutive(lambda kind,**data:events.append((kind,data)),requests.append)
    kwargs=dict(context=dict(wall_s=0.,now_s=0.,map_version=1,resolution=1.,origin_x=0.,origin_y=0.),
        bfs={'cost':{(0,0):0,(0,1):1,(0,2):2},'came_from':{(0,0):None,(0,1):(0,0),(0,2):(0,1)}},
        candidates=CandidateSet({(0,1):5,(0,2):20},{(0,1):(0,1),(0,2):(0,2)},0,0),
        eligible=[(0,1),(0,2)],distance_slack_cells=0)
    return executive,kwargs,events,requests


def test_frontier_size_definition_preserves_candidate_schema():
    from hierarchical_driver import PROMPT, INPUT_SPEC_VERSION, FRONTIER_SIZE_DEFINITION
    assert INPUT_SPEC_VERSION == 'hierarchical-onboard-v3'
    assert FRONTIER_SIZE_DEFINITION == ('frontier_component_cell_count is the number of cells in the frontier component. '
        'It is not free-space area, accessible-area coverage, or a guarantee of traversability.')
    assert FRONTIER_SIZE_DEFINITION in PROMPT
    assert '"cells" means frontier_component_cell_count' in PROMPT
    e,k,events,requests=fixture();e.select(**k)
    assert [c['cells'] for c in requests[0]['candidates']] == [5,20]
    assert all(set(c)=={'id','x_m','y_m','cells'} for c in requests[0]['candidates'])


def test_selected_destination_controls_path_and_coalesces_updates():
    e,k,events,requests=fixture()
    assert ClassicalSelector().select(**k)[0]==(0,1)
    for version in range(10):
        k['context']['map_version']=version
        assert e.select(**k) is None
    assert len(requests)==1
    e.respond(requests[0]['request_id'],'A2')
    cell,path=e.select(**k)
    assert cell==(0,2) and path[-1]==(0,2)
    assert e.state=='executing'


def test_safety_pending_and_delayed_response_cannot_restart():
    e,k,events,requests=fixture();e.select(**k);rid=e.pending['request_id']
    e.safety(False);e.respond(rid,'A2')
    assert e.select(**k) is None and e.state=='stopped'
    e.safety(True);e.cancel('shutdown');e.respond(rid,'A2')
    assert e.select(**k) is None and e.closed


def test_changed_map_rejects_old_destination_then_one_reconsideration():
    e,k,events,requests=fixture();e.select(**k);e.respond(e.pending['request_id'],'A2')
    k['eligible']=[(0,1)];k['context'].update(wall_s=2.,map_version=2)
    assert e.select(**k) is None
    for _ in range(10):e.select(**k)
    assert len(requests)==2 and requests[-1]['reason']=='stale_or_invalid_destination'


@pytest.mark.parametrize('reason,complete',[('quota_exhaustion',False),('completion',True)])
def test_quota_and_completion_latch_stop(reason,complete):
    e,k,events,requests=fixture();e.select(**k);rid=e.pending['request_id']
    if complete:e.cancel(reason,complete=True)
    else:e.respond(rid,error=reason);e.select(**k)
    e.respond(rid,'A1');k['context']['wall_s']=20
    assert e.select(**k) is None and e.closed
    assert e.state==('complete' if complete else 'stopped')


def test_response_timeout_and_model_disabled_by_default(tmp_path):
    e,k,events,requests=fixture();e.select(**k);k['context']['wall_s']=61
    assert e.select(**k) is None and e.closed
    with pytest.raises(RuntimeError,match='model_execution_disabled'):
        PersistentDriver(tmp_path)


def test_real_policy_continuous_tracking_transient_and_persistent_blockage():
    # Reuse original policy fixtures rather than mirror planner implementation.
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('policy_fixture',Path(__file__).resolve().parents[1]/'src/rover_exploration/test/test_exploration_policy.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    e,_,events,requests=fixture()
    policy=f.make_policy(maximum_goal_path_failures=3,maximum_fresh_approaches_per_target=0)
    policy.selector=e;policy.selection_context=dict(wall_s=0.,map_version=1)
    clusters=[{(5,c) for c in range(10,15)},{(8,c) for c in range(10,15)}]
    f.cycle(policy,clusters);assert len(requests)==1
    e.respond(e.pending['request_id'],'A2')
    first=f.cycle(policy,clusters);assert first.goal_assigned and first.path
    for i in range(15):
        assert f.cycle(policy,clusters,now_s=i*.1).path
    assert len(requests)==1
    cell=first.selected_cell
    blocked=[0]*(f.WIDTH*f.HEIGHT);blocked[cell[0]*f.WIDTH+cell[1]]=100
    f.cycle(policy,clusters,planning_data=blocked)
    assert len(requests)==1
    assert f.cycle(policy,clusters).path  # transient blockage clears locally
    policy.selection_context.update(wall_s=4.,map_version=2)
    for _ in range(5):f.cycle(policy,clusters,planning_data=blocked)
    assert len(requests)==2 and requests[-1]['reason']=='bounded_replanning_exhausted'


def test_gate_stops_pending_stale_and_terminal_without_driver():
    from hierarchical_controller import HierarchicalController
    import queue
    from geometry_msgs.msg import Twist
    e,_,_,_=fixture()
    sent=[]
    gate=SimpleNamespace(safety_reason=lambda:None,safe_since=0.,resume_debounce=0.,was_safe=True,
        executive=e,responses=queue.Queue(),policy=SimpleNamespace(complete=False,target=None,recovery_state='idle',counters=SimpleNamespace(goals_reached=0)),
        guarded=(Twist(),__import__('time').monotonic()),idle_cause=None,idle_since=0.,
        event=lambda *a,**k:None,velocity=SimpleNamespace(publish=sent.append),status=SimpleNamespace(publish=lambda m:None))
    gate.guarded[0].linear.x=.15
    HierarchicalController.safety_tick(gate)
    assert sent[-1].linear.x==0  # awaiting selection despite incoming controller motion
    gate.policy.target=SimpleNamespace(goal_world=(1.,0.))
    HierarchicalController.safety_tick(gate);assert sent[-1].linear.x==.15
    gate.safety_reason=lambda:'clock_stall'
    HierarchicalController.safety_tick(gate);assert sent[-1].linear.x==0
    e.cancel('quota_exhaustion');gate.safety_reason=lambda:None
    HierarchicalController.safety_tick(gate);assert sent[-1].linear.x==0


def test_shutdown_with_closed_ros_context_does_not_publish(monkeypatch):
    import hierarchical_controller as adapter
    e,_,_,_=fixture();sent=[]
    import threading
    node=SimpleNamespace(executive=e,velocity=SimpleNamespace(publish=sent.append),shutdown_event=threading.Event(),driver=None)
    monkeypatch.setattr(adapter.rclpy,'ok',lambda:False)
    adapter.HierarchicalController.cancel(node,'shutdown')
    assert e.closed and not sent


def test_obsolete_goal_debounce_and_no_fallback():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('policy_fixture',Path(__file__).resolve().parents[1]/'src/rover_exploration/test/test_exploration_policy.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    e,_,events,requests=fixture();p=f.make_policy(approach_search_radius_m=1.5)
    p.selector=e;p.obsolete_debounce_s=2.;p.selection_context=dict(wall_s=0.,map_version=1)
    clusters=[{(5,c) for c in range(10,15)}]
    f.cycle(p,clusters);e.respond(e.pending['request_id'],'A1');f.cycle(p,clusters)
    assert f.cycle(p,[],now_s=1).path
    assert not f.cycle(p,[],now_s=3.1).path
    assert p.target is None and len(requests)==1 and not p.complete


def test_corrected_localization_gate_uses_measured_velocity_and_tf(monkeypatch):
    from hierarchical_controller import HierarchicalController
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry, OccupancyGrid
    from sensor_msgs.msg import Image, LaserScan
    import time
    import math
    cam=Image();scan=LaserScan();odom=Odometry();grid=OccupancyGrid();tf=TransformStamped()
    for m in [cam,scan,odom,grid,tf]:m.header.stamp.sec=10
    grid.header.frame_id='map';grid.info.origin.orientation.w=1.
    scan.angle_min=-math.pi;scan.angle_increment=math.pi/180;scan.range_min=.1;scan.range_max=10.;scan.ranges=[float('inf')]*360
    tf.transform.rotation.w=1.
    odom.pose.covariance[0]=1e6;odom.pose.covariance[7]=1e6
    now=time.monotonic()
    node=SimpleNamespace(node_time_s=lambda:10.,last_sim=10.,last_clock_wall=now,clock_limit=1.,
        camera=(cam,now),onboard=SimpleNamespace(scan=(scan,now)),odom=(odom,now),sensor_wall_limit=1.,latest_map=grid,
        velocity_variance_limit=.5,tf_buffer=SimpleNamespace(lookup_transform=lambda *a:tf),previous_map_odom=None,map_jump_m=.25,map_jump_rad=.087)
    assert HierarchicalController.safety_reason(node) is None
    odom.twist.covariance[0]=1.
    assert HierarchicalController.safety_reason(node)=='velocity_estimate_covariance'
    odom.twist.covariance[0]=0.;tf.transform.translation.x=1.
    assert HierarchicalController.safety_reason(node)=='localization_jump'
    node.last_clock_wall=now-2.
    assert HierarchicalController.safety_reason(node)=='clock_stall'


def test_hierarchical_velocity_chain_requires_exclusive_resolved_owners():
    from exploration_pilot import controller_ready
    graph=dict(velocity_publishers=['frontier_detector','experiment_supervisor'],command_publishers=['experiment_supervisor'],
        raw_velocity_publishers=['path_follower'],guarded_velocity_publishers=['obstacle_guard'])
    assert controller_ready(graph,'classical',True)
    for bad in [[],['_NODE_NAME_UNKNOWN_'],['obstacle_guard','other']]:
        assert not controller_ready(dict(graph,guarded_velocity_publishers=bad),'classical',True)


def test_snapshot_logs_exact_onboard_images_and_approach_ids(tmp_path):
    import queue
    from nav_msgs.msg import OccupancyGrid
    from sensor_msgs.msg import Image
    from visual_rover_agent.onboard import map_view
    from hierarchical_controller import HierarchicalController
    grid=OccupancyGrid();grid.info.width=5;grid.info.height=5;grid.info.resolution=1.;grid.info.origin.orientation.w=1.;grid.data=[-1]*5+[0]*15+[-1]*5
    _,encoded=map_view(grid,{'state':'missing'})
    camera=Image();camera.width=1;camera.height=1;camera.step=3;camera.encoding='rgb8';camera.data=[10,20,30]
    snapshot={'slam_map':{'frontiers':[],'state':'valid'},'lidar':{'state':'valid'},'estimated_pose':{'state':'valid'}}
    node=SimpleNamespace(camera=(camera,0),latest_map=grid,onboard=SimpleNamespace(snapshot=lambda _: (snapshot.copy(),encoded)),folder=tmp_path,requests=queue.Queue())
    decision={'request_id':1,'map_version':3,'goal_version':0,'candidates':[{'id':'A1','x_m':1.5,'y_m':2.5,'cells':5}]}
    HierarchicalController.request(node,decision)
    payload=node.requests.get_nowait()
    import json
    assert json.loads((tmp_path/'goal_observations.jsonl').read_text())==payload
    assert len(payload['images'])==2 and payload['snapshot']['map_version']==3
    assert payload['snapshot']['candidates'][0]['id']=='A1'
    assert 'ground_truth' not in json.dumps(payload['snapshot'])


def test_twenty_submitted_turn_limit_including_failed_write():
    import io,threading
    d=PersistentDriver.__new__(PersistentDriver)
    d.closed=False;d.turns=0;d.write_lock=threading.Lock();d.log=io.StringIO()
    d.process=SimpleNamespace(stdin=io.StringIO())
    for i in range(20):d.send({'id':i,'method':'turn/start','params':{}})
    with pytest.raises(RuntimeError,match='decision_turn_limit'):
        d.send({'id':21,'method':'turn/start','params':{}})
    assert d.turns==20 and len(d.process.stdin.getvalue().splitlines())==20
    # Cancel/other RPCs do not spend decision turns; application never retries writes.
    d.send({'id':22,'method':'turn/interrupt','params':{}})
    assert d.turns==20
    class Broken:
        def write(self,_):raise OSError('broken pipe')
    d.turns=0;d.process.stdin=Broken()
    with pytest.raises(OSError):d.send({'id':23,'method':'turn/start','params':{}})
    assert d.turns==1


def test_prepared_pair_is_dry_run_only_without_model_opt_in(tmp_path):
    import subprocess,json
    from pathlib import Path
    command=Path(__file__).resolve().parent/'run_hierarchical_comparison'
    output=tmp_path/'pair'
    dry=subprocess.run([str(command),'--dry-run','--output',str(output)],capture_output=True,text=True,check=True)
    commands=json.loads(dry.stdout)
    assert len(commands)==2 and all('600' in c for c in commands)
    assert '--enable-model' not in commands[0] and '--enable-model' in commands[1]
    blocked=subprocess.run([str(command),'--output',str(output)],capture_output=True,text=True)
    assert blocked.returncode==2 and 'model_execution_disabled' in blocked.stderr
    assert not output.exists()


def test_invalid_ros_stop_still_tears_down_owned_and_latches_submission(tmp_path):
    import io,threading
    from exploration_pilot import cleanup_owned_episode
    from rover_experiment import Processes
    owner=Processes(tmp_path)
    # Only a harmless owned sleeping Python child: no ROS/model/simulator.
    import sys
    child=owner.start('driver',[sys.executable,'-c','import time; time.sleep(60)'])
    def invalid_context():raise RuntimeError("publisher's context is invalid")
    marker=tmp_path/'cancel.requested'
    errors=cleanup_owned_episode(owner,marker,[invalid_context])
    assert child.poll() is not None and not owner.owned
    assert owner.cleanup[-1]['group_gone'] and 'context is invalid' in errors[0]
    assert cleanup_owned_episode(owner,marker,[invalid_context]) # idempotent
    assert len(owner.cleanup)==1
    d=PersistentDriver.__new__(PersistentDriver)
    d.closed=False;d.turns=0;d.write_lock=threading.Lock();d.log=io.StringIO()
    d.cancelling=threading.Event();d.cancel_path=marker
    d.process=SimpleNamespace(stdin=io.StringIO())
    with pytest.raises(RuntimeError,match='episode_cancelled'):
        d.send({'method':'turn/start','params':{}})
    assert d.turns==0 and d.process.stdin.getvalue()==''
    marker.unlink();d.latch_cancel()
    with pytest.raises(RuntimeError,match='episode_cancelled'):
        d.send({'method':'turn/start','params':{}})
    assert d.turns==0


def test_controller_close_latches_before_failed_publication(monkeypatch):
    import io,threading,time
    import hierarchical_controller as adapter
    e,_,_,_=fixture();steps=[]
    def fail_publish(_):
        assert steps==['latched']
        raise RuntimeError('invalid ROS context')
    node=SimpleNamespace(executive=e,shutdown_event=threading.Event(),
        driver=SimpleNamespace(latch_cancel=lambda:steps.append('latched'),close=lambda:steps.append('closed')),
        velocity=SimpleNamespace(publish=fail_publish),event=lambda *a,**k:None,
        worker=SimpleNamespace(join=lambda _:steps.append('joined')),idle_cause='executing',
        idle_since=time.monotonic(),log=io.StringIO())
    node.cancel=lambda reason:adapter.HierarchicalController.cancel(node,reason)
    monkeypatch.setattr(adapter.rclpy,'ok',lambda:True)
    adapter.HierarchicalController.close(node)
    adapter.HierarchicalController.close(node)
    assert steps==['latched','closed','joined'] and e.closed and node.shutdown_event.is_set()


def test_cancel_serializes_with_inflight_submission(tmp_path):
    import threading
    from episode_cancellation import cancellation_lock,latch_cancel
    marker=tmp_path/'cancel.requested';entered=threading.Event();finished=threading.Event()
    def cancel():
        entered.set();latch_cancel(marker);finished.set()
    with cancellation_lock(marker):
        thread=threading.Thread(target=cancel);thread.start()
        assert entered.wait(1) and not marker.exists()
        # An already admitted submission finishes before cancellation is latched.
    thread.join(1)
    assert finished.is_set() and marker.exists()
