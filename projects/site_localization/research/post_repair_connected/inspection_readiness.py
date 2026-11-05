"""ROS discovery barrier for inspection-only AMCL checkpoint runs."""
import argparse
import json
from pathlib import Path
import time

import rclpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("site_inspection_startup_readiness")
    start = time.monotonic()
    stable = None
    ready = False
    observed = {}
    try:
        while time.monotonic() - start < 40:
            rclpy.spin_once(node, timeout_sec=.1)
            names = set(node.get_node_names())
            scan = node.get_subscriptions_info_by_topic("/site/scan")
            request = node.get_subscriptions_info_by_topic("/site/camera_request")
            truth = node.get_subscriptions_info_by_topic("/judge/truth")
            observed = dict(nodes=sorted(names), scan_subscribers=[x.node_name for x in scan],
                            request_subscribers=[x.node_name for x in request],
                            truth_subscribers=[x.node_name for x in truth])
            found = ("site_mission" in names and "moved_asset_camera" in names
                     and "site_outcome_observer" in names
                     and any(x.node_name == "site_mission" for x in scan)
                     and any(x.node_name == "moved_asset_camera" for x in request)
                     and any(x.node_name == "site_outcome_observer" for x in truth))
            if found:
                if stable is None:
                    stable = time.monotonic()
                if time.monotonic() - stable >= 2:
                    ready = True
                    break
            else:
                stable = None
    finally:
        args.output.write_text(json.dumps(dict(ready=ready,elapsed_wall_s=time.monotonic()-start,
                                              stable_discovery_s=2,observed=observed,
                                              truth_used=False),indent=2) + "\n")
        node.destroy_node()
        rclpy.shutdown()
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
