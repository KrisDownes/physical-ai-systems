"""Offline gate check on recorded sensors/TF. No ROS publishers or model calls."""
import json
from pathlib import Path
from types import SimpleNamespace
import time
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tf2_ros import Buffer
from hierarchical_controller import HierarchicalController


def replay(bag):
    reader=SequentialReader();reader.open(StorageOptions(uri=str(bag),storage_id='mcap'),ConverterOptions('cdr','cdr'))
    wanted={'/camera/image_raw','/scan','/map','/odometry/filtered','/tf','/tf_static','/clock'}
    types={t.name:get_message(t.type) for t in reader.get_all_topics_and_types() if t.name in wanted}
    latest={};tf=Buffer();samples=[]
    while reader.has_next():
        topic,data,_=reader.read_next()
        if topic not in types:continue
        msg=deserialize_message(data,types[topic]);latest[topic]=msg
        if topic in ('/tf','/tf_static'):
            for transform in msg.transforms:
                (tf.set_transform_static if topic=='/tf_static' else tf.set_transform)(transform,'recording')
        if topic=='/odometry/filtered':samples.append([msg.pose.covariance[i] for i in (0,7,35)])
    clock=latest['/clock'].clock;sim=clock.sec+clock.nanosec/1e9;now=time.monotonic()
    gate=SimpleNamespace(node_time_s=lambda:sim,last_sim=sim,last_clock_wall=now,clock_limit=1.,
        camera=(latest['/camera/image_raw'],now),onboard=SimpleNamespace(scan=(latest['/scan'],now)),
        odom=(latest['/odometry/filtered'],now),sensor_wall_limit=1.,latest_map=latest['/map'],
        velocity_variance_limit=.5,tf_buffer=tf,previous_map_odom=None,map_jump_m=.25,map_jump_rad=.0872664626)
    reason=HierarchicalController.safety_reason(gate)
    return dict(corrected_gate_stop_reason=reason,final_sim_s=sim,
        pose_covariance_final=samples[-1],pose_covariance_max=[max(s[i] for s in samples) for i in range(3)],
        measured_velocity_covariance=[gate.odom[0].twist.covariance[i] for i in (0,35)],
        interpretation='Offline final recorded sensor state only; no physical route execution or simulation rerun.')

if __name__=='__main__':
    import sys
    result=replay(Path(sys.argv[1]));Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
