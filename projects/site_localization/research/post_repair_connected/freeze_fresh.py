"""Freeze the conventional continuous pipeline before new layout preparation."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
OUT = ROOT / "results/fresh_visibility_20261027"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    names = [
        ROOT / "docs/post_repair_connected/fresh_protocol.md",
        ROOT / "research/constraint_repair/slam_params_radius_1p5.yaml",
        ROOT / "vendor/offline_amcl",
        ROOT / "sim/judge_node.py",
    ]
    for directory in (ROOT / "research/continuous_asset", ROOT / "research/autonomous_asset",
                      ROOT / "research/autonomous_asset/reused_rover", ROOT / "research/moved_asset"):
        names.extend(p for p in directory.rglob("*.py") if "__pycache__" not in str(p))
    names.extend(p for p in (ROOT / "research").glob("*.py"))
    names.extend(p for p in (ROOT / "research/post_repair_connected").glob("*.py"))
    names.extend(p for p in (ROOT / "results/asset_calibration_20261023/frozen_bundle").iterdir() if p.is_file())
    names.extend(p for p in (ROOT / "results/moved_asset_20261024/v2/textures").iterdir() if p.is_file())
    names.extend(p for p in (ROOT / "results/connected_layout_20261025/templates").rglob("*") if p.is_file())
    names = sorted(set(names))
    manifest = {
        "role": "Pipeline and source-template freeze before new layout generation/scoring",
        "method": "unchanged 1.5m mapping + nearest or map-only visibility-gain search",
        "seed": 7501, "known_initial_dock": [-2., -2., 0.],
        "diagnostic_gate": "matched_visibility_20261027 wrong-asset pair: earlier detection and valid gain inspection; moved/absent controls pass",
        "whole_mission_contract": "650s/35m, acquisition 180s/15m, inspection 420s/18m, 36 captures, 0 resets/interventions/contacts",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "files": {str(p.relative_to(REPO)): sha(p) for p in names},
        "new_layout_outcomes_unopened": True,
    }
    (OUT / "pipeline_freeze.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUT / "pipeline_freeze.json", len(names), "files")


if __name__ == "__main__":
    main()
