"""Evaluator-only summary of frozen full continuous transfer missions."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research/continuous_asset"))
sys.path.insert(0, str(ROOT / "research/moved_asset"))
from continuous_asset_analysis import truth_pose, csv_pose, error_summary
from buffered_planning import BufferedPlanner
from visibility_search import VisibilitySearch

OUT = ROOT / "results/fresh_inspection_aware_20261028"


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def one(case, arm):
    path = OUT / ("mission_" + case + "_" + arm)
    if not (path / "outcome.json").exists():
        return None
    outcome = load(path / "outcome.json")
    run = load(path / "run.json")
    if not (path / "inspection/task_done.json").exists():
        return dict(case=case, arm=arm, output=str(path), complete_contract=False,
                    infrastructure_failure=outcome.get("infrastructure_failure"),
                    termination=outcome.get("termination"), acquisition=outcome.get("acquisition"))
    done = load(path / "inspection/task_done.json")
    method = rows(path / "inspection/method.jsonl")
    capture_truth = {record["id"]: record for record in rows(path / "inspection/private/sensor_truth.jsonl")}
    truth = truth_pose(path)
    cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth[:, 1:3], axis=0), axis=1))]
    at = lambda stamp: float(np.interp(stamp, truth[:, 0], cumulative))
    handoff = load(path / "handoff.json")
    handoff_time = handoff["estimated_state"]["time"]
    handoff_path = at(handoff_time)
    observations = [event for event in method if event["event"] == "observation"]
    first = next((event for event in observations if 7 in event["vision"].get("observed_ids", [])), None)
    approach = next((event for event in method if event["event"] == "plan" and event["purpose"] == "approach"), None)
    final_id = (done.get("final_observation") or {}).get("id")
    final = next((observation for observation in outcome["all_observations"] if observation["id"] == final_id), None)
    policy = load(path / "inspection/public/policy.json")
    planner = BufferedPlanner(path / "inspection/map.txt")
    selector = VisibilitySearch(planner, path / "inspection/map.txt", policy["search_coordinates"])
    seen = np.zeros(len(selector.points), bool)
    stations = []
    for station in done["visited_search_positions"]:
        mask = selector.visibility(station)
        predicted = float(np.count_nonzero(mask & ~seen) * selector.area)
        seen |= mask
        records = []
        for event in observations:
            if event["purpose"] not in ("old_view", "search") or np.linalg.norm(np.array(event["associated_pose"][:2]) - station) > .15:
                continue
            capture = capture_truth[event["id"]]
            target = next((asset for asset in capture["assets"] if asset["actual_id"] == 7), None)
            records.append(dict(capture=event["id"], time_s=event["sensor_time"],
                                camera_yaw_rad=capture["robot_pose"][2],
                                decoded_ids=event["vision"].get("observed_ids", []),
                                target_rendered=bool(target and target.get("rendered")),
                                target_usable_image=bool(target and target.get("usable_image")),
                                target_diameter_px=None if not target else target.get("face_diameter_px"),
                                target_visibility=None if not target else target.get("visibility")))
        stations.append(dict(station=station,predicted_new_map_free_area_m2=predicted,
                             camera_captures=records,
                             decoded_target_count=sum(7 in record["decoded_ids"] for record in records)))
    frame = handoff["site_from_slam"]
    slam = csv_pose(path / "acquisition/mapper/mapper_tf_pose.csv")
    slam[:, 1:3] += np.array(frame[:2])
    amcl = csv_pose(path / "inspection/estimates.csv")
    first_time = first["sensor_time"] if first else None
    approach_time = approach["time"] if approach else None
    return dict(
        case=case, arm=arm, output=str(path), continuous_mission=True,
        map_sha256=hashlib.sha256((path / "inspection/map.txt").read_bytes()).hexdigest(),
        acquisition_reason=outcome["acquisition"]["reason"],
        acquisition_time_s=outcome["acquisition_time_s"],
        acquisition_distance_m=outcome["acquisition_physical_distance_m"],
        acquisition_known_free_area_m2=outcome["acquisition"].get("known_free_area_m2"),
        acquisition_localization=error_summary(slam,truth),
        handoff_seed_position_error_m=outcome["handoff_seed_position_error_m"],
        handoff_success=outcome["handoff_success"],
        inspection_localization=error_summary(amcl,truth),
        complete_contract=outcome["complete_contract"],
        final_claimed_capture_physically_valid=bool(final and final["valid_physical_inspection"]),
        false_completion=outcome["false_completion"],
        wrong_asset_acceptances=outcome["wrong_asset_acceptances"],
        contacts=outcome["contacts"],clearance_m=outcome["clearance_m"],
        resets=run["resets_after_start"], interventions=run["interventions"],
        all_owned_processes_reaped=run["all_owned_processes_reaped"],
        total_sim_s=outcome["total_sim_s"],total_path_m=outcome["total_path_m"],
        inspection_wheel_path_m=done["wheel_path_m"],
        captures=outcome["observations"], termination=done["reason"],
        first_id7=None if first is None else dict(capture=first["id"],time_s=first_time,
            inspection_physical_path_m=at(first_time)-handoff_path,
            remaining_18m_physical_proxy_m=18-(at(first_time)-handoff_path),
            remaining_420s_s=420-(first_time-handoff_time)),
        approach=None if approach is None else dict(confirmation_s=approach_time,
            planned_map_path_m=None if approach["path"] is None else approach["path"]["length_m"],
            inspection_physical_path_at_confirmation_m=at(approach_time)-handoff_path,
            remaining_18m_physical_proxy_m=18-(at(approach_time)-handoff_path),
            actual_physical_motion_after_confirmation_m=at(done["time"])-at(approach_time)),
        stations=stations,search_choice_logs=[e for e in method if e["event"]=="search_choice"])


def main():
    manifest = load(OUT / "scenario_freeze.json")
    cases = [record for case, arm in manifest["execution_order"] if (record := one(case, arm)) is not None]
    summary = dict(protocol="docs/inspection_aware/protocol.md",
                   scope="One prospective physical layout, 4 case conditions, 12 independently acquired continuous missions",
                   cases=cases,
                   attempted_runs=sum(row["event"]=="attempt_started" for row in rows(OUT / "attempts.jsonl")))
    (OUT / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    for record in cases:
        print(record["case"],record["arm"],record["complete_contract"],
              record.get("termination"),"captures",record.get("captures"),
              "time",round(record.get("total_sim_s",0),1))


if __name__ == "__main__":
    main()
