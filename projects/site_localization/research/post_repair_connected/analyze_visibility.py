"""Evaluator-only summary of the frozen visibility-search development trial."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from continuous_asset_analysis import csv_pose, error_summary, truth_pose

OUT = ROOT / "results/visibility_search_20261027"
BASE = ROOT / "results/post_repair_connected_20261027"
CASES = ("wrong_asset", "moved", "absent")


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(directory, case):
    p = directory / ("mission_" + case)
    result = load(p / "outcome.json")
    run = load(p / "run.json")
    events = rows(p / "inspection/method.jsonl")
    final_id = (load(p / "inspection/task_done.json").get("final_observation") or {}).get("id")
    final = next((o for o in result["all_observations"] if o["id"] == final_id), None)
    truth = truth_pose(p)
    frame = load(p / "handoff.json")["site_from_slam"]
    slam = csv_pose(p / "acquisition/mapper/mapper_tf_pose.csv")
    slam[:, 1:3] += np.array(frame[:2])
    selected = [e for e in events if e["event"] == "search_choice"]
    detected = [
        {"capture": e["id"], "time_s": e["sensor_time"], "ids": e["vision"].get("observed_ids", [])}
        for e in events
        if e["event"] == "observation" and e["vision"].get("observed_ids")
    ]
    return {
        "case": case,
        "complete_contract": result["complete_contract"],
        "inspection_reason": result["inspection_reason"],
        "false_completion": result["false_completion"],
        "wrong_asset_acceptances": result["wrong_asset_acceptances"],
        "contacts": result["contacts"],
        "clearance_m": result["clearance_m"],
        "total_sim_s": result["total_sim_s"],
        "total_path_m": result["total_path_m"],
        "acquisition_time_s": result["acquisition_time_s"],
        "acquisition_distance_m": result["acquisition_physical_distance_m"],
        "inspection_path_m": result["inspection_path_m"],
        "captures": result["observations"],
        "handoff_seed_error_m": result["handoff_seed_position_error_m"],
        "acquisition_localization": error_summary(slam, truth),
        "inspection_localization": error_summary(csv_pose(p / "inspection/estimates.csv"), truth),
        "search_choices": [
            {"time_s": e["time"], "selected": e["selected"], "rule": e["rule"],
             "options": e["options"], "wall_s": e["elapsed_wall_s"]}
            for e in selected
        ],
        "detections": detected,
        "final_observation_id": final_id,
        "final_observation_physically_valid": bool(final and final["valid_physical_inspection"]),
        "resets": run["resets_after_start"],
        "interventions": run["interventions"],
        "all_processes_reaped": run["all_owned_processes_reaped"],
        "infrastructure_failure": result["infrastructure_failure"],
    }


def main():
    executions = rows(OUT / "execution.jsonl")
    assert [e["case"] for e in executions if e["event"] == "start"] == list(CASES)
    assert [e["case"] for e in executions if e["event"] == "finish"] == list(CASES)
    trial = [summarize(OUT, c) for c in CASES]
    nearest = [summarize(BASE, c) for c in CASES]
    assert all(r["resets"] == 0 and r["interventions"] == 0 and r["all_processes_reaped"] for r in trial)
    assert all(not r["infrastructure_failure"] for r in trial)
    summary = {
        "protocol": "docs/post_repair_connected/visibility_trial_protocol.md",
        "scope": "Opened development cases; each mission independently remapped, so not identical acquired-map causal comparison",
        "visibility_gain": trial,
        "nearest_previous": nearest,
        "attempts_started": sum(e["event"] == "attempt_started" for e in rows(OUT / "attempts.jsonl")),
    }
    (OUT / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    for arm, items in (("gain", trial), ("nearest", nearest)):
        for r in items:
            print(arm, r["case"], r["complete_contract"], r["inspection_reason"],
                  "captures", r["captures"], "time", round(r["total_sim_s"], 1),
                  "distance", round(r["total_path_m"], 2),
                  "final_physically_valid", r["final_observation_physically_valid"])


if __name__ == "__main__":
    main()
