"""Offline mock operational audit. Simulator truth is evaluator-only."""
import json
import math
from collections import Counter
from pathlib import Path
from rosbag2_py import SequentialReader,StorageOptions,ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import Buffer,TransformException


def analyze(root):
    root=Path(root);folder=root/'classical'
    events=[json.loads(l) for l in (folder/'executive.jsonl').read_text().splitlines()]
    requested={e['request_id']:e for e in events if e['kind']=='decision_requested'}
    responses={e['request_id']:e for e in events if e['kind']=='selector_response'}
    accepted=[e for e in events if e['kind']=='goal_selected'];arrivals=[e for e in events if e['kind']=='goal_reached']
    reader=SequentialReader();reader.open(StorageOptions(uri=str(folder/'bag'),storage_id='mcap'),ConverterOptions('cdr','cdr'))
    wanted={'/clock','/tf','/tf_static','/planned_path','/odometry/filtered','/ground_truth/odometry','/cmd_vel_raw','/cmd_vel','/hierarchy_cancel','/map'}
    types={t.name:get_message(t.type) for t in reader.get_all_topics_and_types() if t.name in wanted}
    clock=0.;map_stamps=[];raw=[];velocity=[];paths=[];truth=[];odom=[];cancels=[];tf=Buffer(cache_time=Duration(seconds=10000))
    def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec/1e9
    while reader.has_next():
        topic,data,_=reader.read_next()
        if topic not in types:continue
        m=deserialize_message(data,types[topic])
        if topic=='/clock':clock=m.clock.sec+m.clock.nanosec/1e9
        elif topic in ('/tf','/tf_static'):
            for t in m.transforms:(tf.set_transform_static if topic=='/tf_static' else tf.set_transform)(t,'bag')
        elif topic=='/map':map_stamps.append(stamp(m))
        elif topic=='/planned_path':
            if m.poses:paths.append(dict(sim_s=stamp(m),frame=m.header.frame_id,xy=[[p.pose.position.x,p.pose.position.y] for p in m.poses]))
        elif topic in ('/cmd_vel_raw','/cmd_vel'):
            (raw if topic=='/cmd_vel_raw' else velocity).append([clock,m.linear.x,m.angular.z])
        elif topic=='/hierarchy_cancel':cancels.append(dict(sim_s=clock,reason=m.data))
        else:
            p=m.pose.pose.position
            (truth if topic=='/ground_truth/odometry' else odom).append([stamp(m),p.x,p.y])
    trajectory=[]
    for s,_,_ in odom[::5]:
        try:
            t=tf.lookup_transform('map','base_footprint',Time(seconds=s));p=t.transform.translation
            trajectory.append([s,p.x,p.y])
        except TransformException:pass
    selection=[]
    for a in accepted:
        req=requested[a['request_id']];resp=responses.get(a['request_id']);cid=a['candidate']['id']
        supplied=next((c for c in req['candidates'] if c['id']==cid),None)
        arrived=next((e for e in arrivals if e['goal_version']==a['goal_version']),None)
        end=arrived['sim_s'] if arrived else next((x['sim_s'] for x in accepted if x['sim_s']>a['sim_s']),clock)
        route=next((p for p in paths if p['sim_s']>=a['sim_s']-.05 and math.dist(p['xy'][-1],a['accepted_destination_m'])<.001),None)
        pose=arrived.get('estimated_pose') if arrived else None
        arrival_map_stamp=None;arrival_map_pose=None
        if arrived:
            arrival_map_stamp=max(t for t in map_stamps if t<=arrived['sim_s'])
            try:
                t=tf.lookup_transform('map','base_footprint',Time(seconds=arrival_map_stamp));p=t.transform.translation
                arrival_map_pose=[p.x,p.y]
            except TransformException:pass
        between=[e for e in events if e['kind']=='decision_requested' and a['sim_s']<e['sim_s']<end]
        selection.append(dict(request_id=a['request_id'],goal_version=a['goal_version'],selected_id=resp.get('destination_id') if resp else None,
            accepted_id=cid,supplied_candidate=supplied,accepted_destination_m=a['accepted_destination_m'],
            id_controls_destination=bool(resp and resp['destination_id']==cid and supplied),
            route_endpoint_matches=route is not None,route=route,arrival=arrived,
            arrival_distance_m=math.dist(arrival_map_pose,arrived['value']) if arrival_map_pose else None,
            arrival_map_pose=arrival_map_pose,arrival_map_stamp_s=arrival_map_stamp,latest_pose_timer_sample=pose,
            controller_updates=sum(a['sim_s']<=v[0]<=end for v in raw),
            nonzero_controller_updates=sum(a['sim_s']<=v[0]<=end and (abs(v[1])>.001 or abs(v[2])>.001) for v in raw),
            additional_selector_requests_before_arrival_or_next_acceptance=len(between)))
    state_time=Counter()
    for e in events:
        if e['kind']=='state_duration':state_time[e['cause']]+=e['duration_wall_s']
    manifest=json.loads((folder/'manifest.json').read_text());base=json.loads((folder/'results.json').read_text())
    final=velocity[-1][1:] if velocity else None
    result=dict(diagnostic='mock integration, not LLM performance',observations=sum(1 for _ in (folder/'goal_observations.jsonl').open()),
        selector_requests=len(requested),accepted_goals=len(accepted),arrivals=len(arrivals),selections=selection,
        arrival_criterion_m=.25,episode_wall_s=manifest.get('episode_wall_s'),episode_sim_s=manifest.get('episode_sim_s'),
        path_length_truth_m=base.get('path_length_m'),trajectory_frame='map estimated from recorded TF',trajectory=trajectory,
        truth_trajectory_evaluator_only=truth[::5],safety_stop_reasons=[e for e in events if e['kind']=='state' and e['cause'].startswith('safety:')],
        idle_wall_s_by_cause=dict(state_time),selector_waiting_wall_s=state_time['waiting_for_mock'],
        cancellations=cancels,final_velocity=final,cleanup_complete=manifest.get('cleanup_complete'),
        actual_model_calls=0,model_tokens=None,full_exploration_completion=base.get('original_benchmark_pass'),
        post_cancel_nonzero_velocity_samples=sum(v[0]>cancels[0]['sim_s']+.15 and (abs(v[1])>.001 or abs(v[2])>.001) for v in velocity) if cancels else None)
    result['select_move_arrive_stop_pass']=bool(any(s['arrival'] and s['id_controls_destination'] and s['route_endpoint_matches'] and s['arrival_distance_m'] is not None and s['arrival_distance_m']<=.25 and s['controller_updates']>1 and s['additional_selector_requests_before_arrival_or_next_acceptance']==0 for s in selection) and base.get('path_length_m',0)>.25 and final==[0.,0.] and result['cleanup_complete'])
    (root/'operational_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,6))
    for s in selection:
        if s['route']:
            xy=s['route']['xy'];ax.plot([p[0] for p in xy],[p[1] for p in xy],alpha=.7,label=f"Goal {s['goal_version']} {s['accepted_id']} route")
            ax.scatter(*s['accepted_destination_m'],marker='x',s=70)
    if trajectory:ax.plot([p[1] for p in trajectory],[p[2] for p in trajectory],'k',linewidth=2,label='Recorded estimated trajectory')
    ax.axis('equal');ax.grid();ax.legend();ax.set(xlabel='Online map x (m)',ylabel='Online map y (m)',title='Mock route integration — not model performance')
    fig.tight_layout();fig.savefig(root/'route-trajectory.png',dpi=140);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['selector_requests','accepted_goals','arrivals','path_length_truth_m','final_velocity','select_move_arrive_stop_pass']},indent=2))
    return result

if __name__=='__main__':
    import sys
    analyze(sys.argv[1])
