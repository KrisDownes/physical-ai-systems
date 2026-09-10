"""Independent raw-map benchmark measurement; never selects a goal or path."""
import json
import math
import time
from pathlib import Path
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, Bool
from rover_exploration.result_contract import RESULT_KEYS_V2
from rclpy.qos import QoSProfile, DurabilityPolicy
from rover_exploration.mission_evaluator import count_frontier_components


def attach(monitor, publish_completion=True):
    state_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    monitor.complete_pub = monitor.create_publisher(Bool, '/exploration_complete', state_qos)
    monitor.result_pub = monitor.create_publisher(String, '/exploration_result', state_qos)
    monitor.completion_announced = False
    def announce():
        monitor.complete_pub.publish(Bool(data=monitor.completion_announced))
    if publish_completion:
        monitor.create_timer(.5, announce)
    monitor.coverage = None
    monitor.empty_since = None
    monitor.completed = False
    monitor.timing_events = []
    def timing(msg):
        event = json.loads(msg.data)
        monitor.timing_events.append(event)
        monitor.event('timing', event)
    def mapping(msg):
        total = msg.info.width * msg.info.height
        if not total:
            return
        cells, sizes = count_frontier_components(msg.data, msg.info.width, msg.info.height)
        sim = monitor.latest.get('truth', (0,))[0]
        unresolved = sum(n >= 5 for n in sizes)
        if unresolved:
            monitor.empty_since = None
        elif monitor.empty_since is None:
            monitor.empty_since = sim
        monitor.completed = monitor.empty_since is not None and sim-monitor.empty_since >= 8
        monitor.coverage = dict(known_map_percent=100*sum(v != -1 for v in msg.data)/total,
            denominator_cells=total, frontier_cells=cells, unresolved_components=unresolved,
            completed=monitor.completed, sim_s=sim)
        monitor.event('coverage', monitor.coverage)
    monitor.create_subscription(OccupancyGrid, '/map', mapping, 10)
    monitor.create_subscription(String, '/agent_timing', timing, 100)


def summarize(folder):
    events = [json.loads(l) for l in (folder/'events.jsonl').read_text().splitlines()]
    timing = [e['data'] for e in events if e['kind']=='timing']
    observations = {e['observation_id']:e for e in timing if e['kind']=='observation_ready'}
    poses = [e for e in events if e['kind']=='truth']
    rows=[]
    for request in (e for e in timing if e['kind']=='request'):
        group=[e for e in timing if e.get('id')==request['id']]
        first=lambda kind: next((e for e in group if e['kind']==kind),None)
        pub,rec,accept=first('command_published'),first('executor_received'),first('executor_accepted')
        velocity=next((e for e in group if e['kind']=='velocity_published' and any(e['velocity'])),None)
        terminal=next((e for e in group if e['kind']=='executor_status' and e['state'] in ('succeeded','aborted','rejected')),None)
        zero=next((e for e in group if e['kind']=='velocity_published' and not any(e['velocity']) and (velocity is None or e['wall_s']>velocity['wall_s'])),None)
        obs=observations.get(request.get('observation_id'))
        before=next((e for e in reversed(poses) if velocity and e['wall_s']<=velocity['wall_s']),None)
        motion=None
        if before and velocity:
            for e in poses:
                if e['wall_s']<velocity['wall_s']: continue
                if terminal and e['wall_s']>terminal['wall_s']: break
                a,b=e['data'],before['data']
                if math.hypot(a['x']-b['x'],a['y']-b['y'])>=.001 or abs(math.atan2(math.sin(a['yaw']-b['yaw']),math.cos(a['yaw']-b['yaw'])))>=.001:
                    motion=e; break
        stopped = None
        if zero:
            stable = []
            for previous,current in zip(poses,poses[1:]):
                if previous['wall_s'] < zero['wall_s']: continue
                # Restrict to this command's quiet period, not a later stop.
                later = next((e['wall_s'] for e in timing if e['kind']=='request' and e['wall_s']>zero['wall_s']), float('inf'))
                if current['wall_s'] >= later: break
                a,b=previous['data'],current['data']; dt=b['t']-a['t']
                if dt <= 0: continue
                linear=math.hypot(b['x']-a['x'],b['y']-a['y'])/dt
                angular=abs(math.atan2(math.sin(b['yaw']-a['yaw']),math.cos(b['yaw']-a['yaw'])))/dt
                if linear < .005 and angular < .005:
                    stable.append(current)
                    if len(stable)>=3:
                        stopped=stable[0];break
                else: stable=[]
        delta=lambda a,b: a['wall_s']-b['wall_s'] if a and b else None
        row=dict(id=request['id'],action=request['action'],image_preparation_s=obs['preparation_s'] if obs else None,
            observation_ready_to_request_s=delta(request,obs),mcp_processing_s=delta(pub,request),ros_transport_s=delta(rec,pub),
            acceptance_to_velocity_s=delta(velocity,accept),velocity_to_motion_s=delta(motion,velocity),
            observation_to_motion_s=motion['wall_s']-obs['received_wall_s'] if motion and obs else None,
            command_delivery_s=delta(velocity,pub),
            feedback_delay_s=next((e['wall_s']-terminal['wall_s'] for e in timing if terminal and e['kind']=='observation_ready' and e['sim_s']>terminal['sim_s'] and e['wall_s']>=terminal['wall_s']),None),
            execution_s=delta(terminal,accept),stop_request_to_zero_s=delta(zero,request) if request['action']=='stop' else None,
            post_action_frame_wait_s=next((e['wall_s']-terminal['wall_s'] for e in timing if terminal and e['kind']=='camera_received' and e['wall_s']>terminal['wall_s'] and e['sim_s']>terminal['sim_s']),None),
            physical_stopping_s=delta(stopped,zero), events=group)
        rows.append(row)
    def stats(values):
        values=sorted(values)
        def q(p):
            rank=(len(values)-1)*p; lo=int(rank); hi=math.ceil(rank)
            return values[lo]+(values[hi]-values[lo])*(rank-lo)
        return dict(n=len(values),median=q(.5),p95=q(.95),maximum=max(values)) if values else dict(n=0,median=None,p95=None,maximum=None)
    distributions={k:stats([r[k] for r in rows if r[k] is not None]) for k in rows[0] if k.endswith('_s')} if rows else {}
    ticks=[e['wall_s'] for e in timing if e['kind']=='control_tick']
    intervals=[b-a for a,b in zip(ticks,ticks[1:])]
    sampling=[b['wall_s']-a['wall_s'] for a,b in zip(poses,poses[1:])]
    coverage=[dict(wall_s=e['wall_s'],**e['data']) for e in events if e['kind']=='coverage']
    vel=[e for e in timing if e['kind']=='velocity_published']
    moving=sum(b['wall_s']-a['wall_s'] for a,b in zip(vel,vel[1:]) if any(a['velocity']))
    waiting=sum(r['observation_ready_to_request_s'] or 0 for r in rows)
    report=dict(distributions=distributions,control_interval_s=stats(intervals),control_rate_hz=1/(sum(intervals)/len(intervals)) if intervals else None,
        truth_sampling_wall_s=stats(sampling),motion_threshold_m=.001,motion_threshold_rad=.001, stopping_linear_threshold_mps=.005, stopping_angular_threshold_radps=.005, stopping_consecutive_samples=3,
        moving_command_wall_s=moving,observation_ready_to_request_total_s=waiting,
        unavailable=['MCP serialization/return delivery and harness receipt; ready is a pre-return proxy', 'pure model inference', 'contact collisions'],
        final_coverage=coverage[-1] if coverage else None)
    (folder/'latency.json').write_text(json.dumps(report,indent=2))
    (folder/'action_timeline.json').write_text(json.dumps(rows,indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(3,1,figsize=(12,11))
        for key in distributions:
            axs[0].plot([r[key] if r[key] is not None else float('nan') for r in rows],label=key)
        if rows: axs[0].legend(fontsize=6)
        axs[0].set(ylabel='Wall seconds',xlabel='Action',title='Measured latency (ready is pre-return proxy)')
        if coverage: axs[1].plot([e['wall_s']-events[0]['wall_s'] for e in coverage],[e['known_map_percent'] for e in coverage])
        axs[1].set(ylabel='Known grid % (diagnostic)',xlabel='Elapsed wall seconds')
        axs[2].bar(['Command moving','Observation ready to request'],[moving,waiting]); axs[2].set(ylabel='Wall seconds')
        fig.tight_layout(); fig.savefig(folder/'measurements.png'); plt.close(fig)
    except ImportError as error:
        (folder/'plot_error.txt').write_text(str(error))
    return report


def announce_completion(monitor):
    value = {k: 0 for k in RESULT_KEYS_V2}
    value.update(schema_version=2, completed=True, completion_time_s=monitor.latest['truth'][0],
        outcome='success', blocked_reason=None, frontier_cells=monitor.coverage['frontier_cells'],
        geometric_frontier_cells=monitor.coverage['frontier_cells'])
    monitor.completion_announced=True
    monitor.complete_pub.publish(Bool(data=True))
    monitor.result_pub.publish(String(data=json.dumps(value)))


def benchmark_bag(folder, classical=False):
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rover_exploration.mission_evaluator import REQUIRED_TOPICS, evaluate_mission
    reader=SequentialReader()
    reader.open(StorageOptions(uri=str(folder/'bag'), storage_id='mcap'), ConverterOptions('cdr','cdr'))
    types={t.name:get_message(t.type) for t in reader.get_all_topics_and_types() if t.name in REQUIRED_TOPICS}
    collected={t:[] for t in REQUIRED_TOPICS}
    while reader.has_next():
        topic,data,stamp=reader.read_next()
        if topic in types: collected[topic].append((stamp,deserialize_message(data,types[topic])))
    result,passed,reasons=evaluate_mission(collected)
    result['agent_mode_absent_control_topics']=[] if classical else ['/planned_path','/cmd_vel_raw','/recovery_request']
    result['measurement_adapter']='Classical native messages' if classical else 'Original evaluator on original messages; disabled classical control streams are empty, not synthesized.'
    (folder/'original_benchmark.json').write_text(json.dumps(result,indent=2))
    return result
