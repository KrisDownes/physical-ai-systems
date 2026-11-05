"""ROS discovery barrier before physics; no robot state or truth input."""
import argparse
import json
from pathlib import Path
import time
import rclpy


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rclpy.init();node=rclpy.create_node('site_startup_readiness')
    start=time.monotonic();stable=None;ready=False;subscribers=[];publishers=[]
    try:
        while time.monotonic()-start<40:
            rclpy.spin_once(node,timeout_sec=.1)
            subscribers=node.get_subscriptions_info_by_topic('/pose')
            publishers=node.get_publishers_info_by_topic('/pose')
            found=any(s.node_name=='site_mission' for s in subscribers) and any(v.node_name=='slam_toolbox' for v in publishers)
            if found:
                if stable is None:stable=time.monotonic()
                if time.monotonic()-stable>=2:ready=True;break
            else:stable=None
    finally:
        a.output.write_text(json.dumps(dict(ready=ready,elapsed_wall_s=time.monotonic()-start,
            stable_discovery_s=2,subscriber_nodes=[s.node_name for s in subscribers],
            publisher_nodes=[v.node_name for v in publishers],truth_used=False),indent=2)+'\n')
        node.destroy_node();rclpy.shutdown()
    if not ready:raise SystemExit(2)


if __name__=='__main__':main()
