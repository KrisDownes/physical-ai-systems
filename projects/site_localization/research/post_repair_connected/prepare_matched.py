"""Build evaluator-controlled, identical inspection handoffs from opened acquisitions.

The physical restart pose is private simulation setup. The policy gets the saved
estimated seed and acquired map only. This creates a diagnostic reset, not a
continuous mission.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "research"))
from continuous_asset_analysis import truth_pose

BASE = ROOT / "results/post_repair_connected_20261027"
OUT = ROOT / "results/matched_visibility_20261027"
CASES = ("wrong_asset", "moved", "absent")


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    templates = OUT / "templates"
    templates.mkdir()
    template_audit = {}
    for case in CASES:
        source = BASE / ("mission_" + case)
        handoff = load(source / "handoff.json")
        stamp = handoff["estimated_state"]["time"]
        truth = truth_pose(source)
        physical = [float(np.interp(stamp, truth[:, 0], truth[:, j])) for j in (1, 2, 3)]
        assert np.isfinite(physical).all()
        case_dir = templates / case
        case_dir.mkdir()
        (case_dir / "private").mkdir()
        inspected = case_dir / "inspection"
        inspected.mkdir()
        (inspected / "public").mkdir()
        (inspected / "private").mkdir()
        for name in ("map.txt", "controller_config.json"):
            shutil.copyfile(source / "inspection" / name, inspected / name)
        for name in ("vision.json",):
            shutil.copyfile(source / "inspection/public" / name, inspected / "public" / name)
        for name in ("scene.json", "evaluation.json"):
            shutil.copyfile(source / "inspection/private" / name, inspected / "private" / name)
        world = (source / "private/world.sdf").read_text()
        new_pose = " ".join(f"{value:.12g}" for value in (*physical[:2], 0., 0., 0., physical[2]))
        pattern = r'(<model name="rover">\s*<pose>)[^<]+(</pose>)'
        world, count = re.subn(pattern, lambda match: match.group(1) + new_pose + match.group(2), world)
        assert count == 1, (case, count)
        (case_dir / "private/world.sdf").write_text(world)
        for arm in ("nearest", "gain"):
            policy = load(source / "inspection/public/policy.json")
            assert policy["handoff_time"] == stamp
            assert abs(policy["max_sim_s"] - stamp - 420) < 1e-6
            policy["handoff_time"] = 0.
            policy["max_sim_s"] = 420.
            if arm == "gain":
                policy["search_selector"] = "visibility_gain"
            else:
                assert "search_selector" not in policy
            save(inspected / "public" / ("policy_" + arm + ".json"), policy)
        evaluation = load(inspected / "private/evaluation.json")
        assert evaluation["max_sim_s"] >= 420
        evaluation["max_sim_s"] = 420.
        save(inspected / "private/evaluation.json", evaluation)
        handoff["original_estimated_time"] = stamp
        handoff["estimated_state"]["time"] = 0.
        handoff["estimated_state"]["pose_time"] = 0.
        handoff["checkpoint_physical_pose_from_evaluator_only"] = True
        handoff["simulator_restarted"] = True
        save(case_dir / "handoff.json", handoff)
        save(case_dir / "checkpoint_private.json", {
            "source": str(source), "original_handoff_time_s": stamp,
            "evaluator_physical_restart_pose": physical,
            "method_initial_estimate": handoff["amcl_initial_pose"],
            "method_initial_std_xy_m": handoff["amcl_initial_std_xy_m"],
            "method_initial_std_yaw_deg": handoff["amcl_initial_std_yaw_deg"],
            "truth_used_to_construct_simulator_checkpoint": True,
            "truth_provided_to_policy": False,
        })
        template_audit[case] = {
            "source_map_sha256": sha(source / "inspection/map.txt"),
            "checkpoint_map_sha256": sha(inspected / "map.txt"),
            "source_world_sha256": sha(source / "private/world.sdf"),
            "checkpoint_physical_pose": physical,
            "method_seed_pose": handoff["amcl_initial_pose"],
            "policy_differences": ["gain adds search_selector=visibility_gain"],
        }
        assert template_audit[case]["source_map_sha256"] == template_audit[case]["checkpoint_map_sha256"]
    method_files = [
        ROOT / "research/continuous_asset/inspection.py",
        ROOT / "research/continuous_asset/visibility_search.py",
        ROOT / "research/continuous_asset/buffered_planning.py",
        ROOT / "research/continuous_asset/inspection_judge.py",
        ROOT / "research/controller.py",
        ROOT / "research/moved_asset/planning.py",
        ROOT / "research/moved_asset/camera.py",
        ROOT / "research/moved_asset/render.py",
        ROOT / "research/autonomous_asset/vision.py",
        ROOT / "sim/judge_node.py",
        ROOT / "vendor/offline_amcl",
        ROOT / "research/post_repair_connected/prepare_matched.py",
        ROOT / "research/post_repair_connected/run_matched.py",
        ROOT / "docs/post_repair_connected/matched_protocol.md",
    ]
    method_files += [p for p in templates.rglob("*") if p.is_file()]
    manifest = {
        "source_cohort": str(BASE),
        "cases": list(CASES),
        "execution_order": [["wrong_asset", "nearest"], ["wrong_asset", "gain"],
                            ["moved", "gain"], ["moved", "nearest"],
                            ["absent", "nearest"], ["absent", "gain"]],
        "diagnostic_reset_per_arm": 1,
        "continuous_mission_claim": False,
        "wall_guard_s": 800,
        "template_audit": template_audit,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "files": {str(p.relative_to(REPO)): sha(p) for p in method_files},
        "outcomes_unopened_at_freeze": True,
    }
    save(OUT / "manifest.json", manifest)
    print(OUT / "manifest.json")


if __name__ == "__main__":
    main()
