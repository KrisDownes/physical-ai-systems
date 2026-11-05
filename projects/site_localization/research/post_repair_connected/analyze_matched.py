"""Evaluator-only checkpoint comparison; never imported by the robot policy."""
import json
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research/continuous_asset"))
sys.path.insert(0, str(ROOT / "research/moved_asset"))
from continuous_asset_analysis import truth_pose
from buffered_planning import BufferedPlanner
from visibility_search import VisibilitySearch

OUT = ROOT / "results/matched_visibility_20261027"


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def one(case, arm, suffix=""):
    path = OUT / ("mission_" + case + "_" + arm + suffix)
    if not (path / "inspection/outcome.json").exists():
        return None
    task = load(path / "inspection/outcome.json")
    done = load(path / "inspection/task_done.json")
    run = load(path / "run.json")
    log = rows(path / "inspection/method.jsonl")
    sensor = {capture["id"]: capture for capture in rows(path / "inspection/private/sensor_truth.jsonl")}
    truth = truth_pose(path)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth[:, 1:3], axis=0), axis=1))]
    path_at = lambda stamp: float(np.interp(stamp, truth[:, 0], distance))
    observation = [event for event in log if event["event"] == "observation"]
    first = next((event for event in observation if 7 in event["vision"].get("observed_ids", [])), None)
    ready = next((event for event in log if event["event"] == "localization_ready"), None)
    plans = [event for event in log if event["event"] == "plan" and event["purpose"] == "approach"]
    approach = plans[0] if plans else None
    checkpoint = load(path / "checkpoint_private.json")
    restart_error = float(np.linalg.norm(truth[0, 1:3] - checkpoint["evaluator_physical_restart_pose"][:2]))
    planner = BufferedPlanner(path / "inspection/map.txt")
    policy = load(path / "inspection/public/policy.json")
    visibility = VisibilitySearch(planner, path / "inspection/map.txt", policy["search_coordinates"])
    seen = np.zeros(len(visibility.points), bool)
    station_records = []
    for station in done["visited_search_positions"]:
        station = [float(v) for v in station]
        mask = visibility.visibility(station)
        predicted = float(np.count_nonzero(mask & ~seen) * visibility.area)
        seen |= mask
        captures = []
        for event in observation:
            if event["purpose"] not in ("old_view", "search"):
                continue
            capture = sensor[event["id"]]
            # Station assignment follows the robot's commanded/map-frame state;
            # physical offset is evaluated separately and may exceed 0.15 m.
            if np.linalg.norm(np.array(event["associated_pose"][:2]) - station) > .15:
                continue
            target = next((asset for asset in capture["assets"] if asset["actual_id"] == 7), None)
            captures.append({
                "id": event["id"], "time_s": event["sensor_time"],
                "camera_yaw_rad": capture["robot_pose"][2],
                "physical_station_offset_m": float(np.linalg.norm(np.array(capture["robot_pose"][:2]) - station)),
                "decoded_ids": event["vision"].get("observed_ids", []),
                "target_rendered": bool(target and target.get("rendered")),
                "target_usable_image": bool(target and target.get("usable_image")),
                "target_face_diameter_px": None if not target else target.get("face_diameter_px"),
                "target_visibility_fractions": None if not target else target.get("visibility"),
                "target_distance_m": None if not target else target.get("distance_m"),
            })
        station_records.append({"station": station, "predicted_new_map_free_area_m2": predicted,
                                "captures": captures, "decoded_target_count": sum(7 in c["decoded_ids"] for c in captures),
                                "actual_usable_target_images": sum(c["target_usable_image"] for c in captures)})
    final_id = (done.get("final_observation") or {}).get("id")
    final = next((m for m in task["observations"] if m["id"] == final_id), None)
    first_time = first["sensor_time"] if first else None
    approach_time = approach["time"] if approach else None
    return {
        "case": case, "arm": arm, "output": str(path),
        "diagnostic_reset": True, "continuous_mission": False,
        "physical_restart_position_error_m": restart_error,
        "first_localization_ready": None if ready is None else {
            "time_s": ready["time"], "estimated_pose": ready["estimate"],
            "position_error_m": float(np.linalg.norm(np.array(ready["estimate"][:2]) -
                np.array([np.interp(ready["time"], truth[:, 0], truth[:, 1]),
                          np.interp(ready["time"], truth[:, 0], truth[:, 2])]))),
        },
        "acquired_map_sha256": hashlib.sha256((path / "inspection/map.txt").read_bytes()).hexdigest(),
        "success_claim": task["success_claim"],
        "complete_physical_inspection": bool(task["success_claim"] and task["safe_navigation"]
                                             and final and final["valid_physical_inspection"]
                                             and run["stop_reason"] == "inspection_finished"),
        "final_claimed_capture_physically_valid": bool(final and final["valid_physical_inspection"]),
        "false_completion": bool(task["success_claim"] and not(final and final["valid_physical_inspection"]
                                                                 and task["safe_navigation"])),
        "wrong_asset_acceptances": task["wrong_asset_acceptances"],
        "contacts": task["contact_events"], "minimum_clearance_m": task["minimum_clearance_m"],
        "termination_reason": task["termination_reason"], "captures": task["captures"],
        "sim_time_s": task["simulation_time_s"], "wheel_path_m": task["wheel_path_m"],
        "physical_path_m": task["physical_path_m"],
        "first_target_detection": None if first is None else {
            "capture_id": first["id"], "time_s": first_time,
            "physical_path_m": path_at(first_time),
            "remaining_18m_physical_proxy_m": 18 - path_at(first_time),
            "remaining_420s_s": 420 - first_time,
        },
        "approach": None if approach is None else {
            "confirmation_time_s": approach_time,
            "planned_map_path_m": approach["path"]["length_m"] if approach["path"] else None,
            "physical_path_at_confirmation_m": path_at(approach_time),
            "remaining_18m_physical_proxy_m": 18 - path_at(approach_time),
            "actual_physical_motion_after_confirmation_m": path_at(done["time"]) - path_at(approach_time),
        },
        "station_observations": station_records,
        "search_choice_logs": [event for event in log if event["event"] == "search_choice"],
        "all_owned_processes_reaped": run["all_owned_processes_reaped"],
    }


def main():
    order = (("wrong_asset", "nearest", "_infra_retry02"), ("wrong_asset", "gain", ""),
             ("moved", "gain", ""), ("moved", "nearest", ""),
             ("absent", "nearest", ""), ("absent", "gain", ""))
    cases = [record for case, arm, suffix in order if (record := one(case, arm, suffix)) is not None]
    output = {"protocol": "docs/post_repair_connected/matched_protocol.md",
              "scope": "Inspection-only diagnostic restarts from the same per-case physical/map/estimated handoff",
              "completed_arms": cases,
              "attempt_count_including_infrastructure": sum(line.get("event") == "attempt_started"
                                                       for line in rows(OUT / "attempts.jsonl"))}
    (OUT / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        first = case["first_target_detection"] or {}
        print(case["case"], case["arm"], case["complete_physical_inspection"],
              case["termination_reason"], "first", round(first["time_s"], 1) if first else None,
              "path", round(first["physical_path_m"], 2) if first else None,
              "captures", case["captures"])


if __name__ == "__main__":
    main()
