import json
from pathlib import Path
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def verify(folder):
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(folder/'bag'),storage_id=''),rosbag2_py.ConverterOptions('',''))
    types={t.name:get_message(t.type) for t in reader.get_all_topics_and_types()}
    counts={}; frames=set(); statuses={}; commands=set(); final_pose=None; velocity=None
    while reader.has_next():
     topic,data,_=reader.read_next();counts[topic]=counts.get(topic,0)+1
     if topic not in {'/camera/image_raw','/agent_status','/agent_command','/ground_truth/odometry','/cmd_vel'}:continue
     m=deserialize_message(data,types[topic])
     if topic=='/camera/image_raw':frames.add(round(m.header.stamp.sec+m.header.stamp.nanosec/1e9,6))
     elif topic=='/agent_status':
      s=json.loads(m.data);statuses.setdefault(s['id'],[]).append(s)
     elif topic=='/agent_command':commands.add(json.loads(m.data)['id'])
     elif topic=='/ground_truth/odometry':final_pose=[m.pose.pose.position.x,m.pose.pose.position.y]
     elif topic=='/cmd_vel':velocity=[m.linear.x,m.angular.z]
    audit=json.loads((folder/'driver_audit.json').read_text())
    observed=[o['captured_sim_time_s'] for o in audit['observations']]
    movements=[]
    for call in audit['calls']:
     if call.get('tool') not in {'drive','turn'}:continue
     for c in (call.get('result') or {}).get('content',[]):
      if c['type']=='text':
       try:r=json.loads(c['text'])
       except ValueError:continue
       if 'command_id' in r:movements.append(r['command_id'])
    result={'episode':str(folder),'counts':counts,'final_pose_xy':final_pose,'final_velocity':velocity,
    'all_observation_frames_in_bag':all(round(t,6) in frames for t in observed),
    'movement_commands_missing_from_bag':[i for i in movements if i not in commands],
    'movement_terminal_statuses_missing_from_bag':[i for i in movements if not any(s['state'] in {'succeeded','aborted','rejected'} for s in statuses.get(i,[]))]}
    (folder/'bag_verification.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    results = [verify(folder) for folder in sorted(args.root.glob('*/episode-*'))
               if (folder/'bag/metadata.yaml').exists() and (folder/'driver_audit.json').exists()]
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
