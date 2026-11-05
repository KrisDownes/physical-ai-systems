"""Prospective connected service-suite scenarios and evaluator-only preflight."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "research/moved_asset"))
sys.path.insert(0, str(ROOT / "research/connected_layout"))
from planning import GridPlanner
from render import render
from prepare import floor_grid

OUT = ROOT / "results/fresh_inspection_aware_20261028"
SOURCE = ROOT / "results/connected_layout_20261025/templates"
CASES = ("visible", "moved", "wrong_asset", "absent")


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry():
    return [
        [-3.2, 0., .15, 6.55, 1.], [3.2, 0., .15, 6.55, 1.],
        [0., -3.2, 6.55, .15, 1.], [0., 3.2, 6.55, .15, 1.],
        [.65, -2.5, .18, 1.4, 1.], [.65, 1.45, .18, 3.5, 1.],
        [-2.175, .55, 2.05, .18, 1.],
    ]


def main():
    pipeline_freeze = load(OUT / "pipeline_freeze.json")
    assert all(sha(REPO / name) == expected for name, expected in pipeline_freeze["files"].items())
    templates = OUT / "templates"
    templates.mkdir(exist_ok=False)
    oracle = OUT / "evaluator_preflight"
    oracle.mkdir()
    boxes = geometry()
    axis, points, free, connected, clearance = floor_grid(boxes)
    occupied = (clearance == 0).reshape(free.shape)
    truth_grid = oracle / "truth_grid.txt"
    with truth_grid.open("w") as stream:
        stream.write(f"{len(axis)} {len(axis)} .05 {axis[0]} {axis[0]}\n")
        np.savetxt(stream, np.where(occupied, 1, -1), fmt="%d")
    planner = GridPlanner(truth_grid, .52)
    planned = {"visible": [0., -1.5, 0.], "moved": [2.45, 1.4, 0.],
               "wrong_asset": [2.45, 1.4, 0.], "absent": None}
    records = []
    for case in CASES:
        source = SOURCE / case
        dst = templates / case
        dst.mkdir()
        (dst / "private").mkdir()
        shutil.copytree(source / "public", dst / "public")
        shutil.copyfile(source / "controller_config.json", dst / "controller_config.json")
        scene = load(source / "private/scene.json")
        scene["obstacles"] = boxes
        scene["assets"] = []
        for identity, pose in ((7, planned[case]),
                               (8, [0., -1.5, 0.] if case in ("wrong_asset", "absent") else None)):
            if pose is not None:
                scene["assets"].append(dict(id=identity,pose=pose,
                    texture=str(ROOT / f"results/moved_asset_20261024/v2/textures/asset_{identity}.png")))
        save(dst / "private/scene.json", scene)
        evaluation = load(source / "private/evaluation.json")
        evaluation.update(case=case, target_present=planned[case] is not None,
                          physical_obstacles=[box[:4] for box in boxes])
        save(dst / "private/evaluation.json", evaluation)
        tree = ET.parse(source / "private/world.sdf")
        world = tree.getroot().find("world")
        sample = copy.deepcopy(world.find("model[@name='partition_0']"))
        assert sample is not None and world.find("model[@name='rover']") is not None
        for model in list(world.findall("model")):
            if model.get("name") not in ("floor", "rover"):
                world.remove(model)
        for i, (x, y, sx, sy, height) in enumerate(boxes):
            model = copy.deepcopy(sample)
            model.set("name", f"partition_{i}")
            model.find("pose").text = f"{x} {y} {height/2} 0 0 0"
            for size in model.findall(".//box/size"):
                size.text = f"{sx} {sy} {height}"
            world.append(model)
        for asset in scene["assets"]:
            x, y, yaw = asset["pose"]
            model = ET.fromstring(
                f'<model name="asset_panel_{asset["id"]}"><static>true</static>'
                f'<pose>{x} {y} .8 0 0 {yaw}</pose><link name="panel">'
                '<collision name="panel"><geometry><box><size>.01 .667 .4</size>'
                '</box></geometry></collision></link></model>')
            world.append(model)
        tree.write(dst / "private/world.sdf", encoding="unicode")
        nearest_policy = load(dst / "public/policy.json")
        assert "search_selector" not in nearest_policy
        record = dict(case=case, target_present=planned[case] is not None,
                      all_footprint_free_space_connected=bool(np.all(connected[free])),
                      connected_floor_area_m2=float(connected.sum() * .05**2))
        if planned[case] is not None:
            x, y, yaw = planned[case]
            witness = [x - 1.1 * math.cos(yaw), y - 1.1 * math.sin(yaw), yaw]
            path = planner.path([-2., -2.], witness)
            image, rendered = render(witness, scene)
            target = next(asset for asset in rendered["assets"] if asset["actual_id"] == 7)
            cv2.imwrite(str(oracle / f"{case}_witness.png"), image)
            record.update(evaluator_only_witness=witness,
                          evaluator_only_witness_path_m=None if path is None else path["length_m"],
                          witness_usable_image=target["usable_image"],
                          witness_target_pixels=target.get("face_diameter_px"))
            assert path is not None and target["usable_image"], record
        assert record["all_footprint_free_space_connected"], record
        records.append(record)
        gain_dir = templates / (case + "_gain")
        nearest_dir = templates / (case + "_nearest")
        aware_dir = templates / (case + "_aware")
        shutil.copytree(dst, gain_dir)
        save(gain_dir / "public/policy.json", dict(nearest_policy, search_selector="visibility_gain"))
        shutil.copytree(dst, aware_dir)
        save(aware_dir / "public/policy.json", dict(nearest_policy, search_selector="camera_belief"))
        dst.rename(nearest_dir)
    save(oracle / "feasibility.json", dict(scope="Evaluator-only geometry, path and renderer preflight",
                                           method_inputs_exclude_truth=True, cases=records,
                                           physical_footprint_radius_m=.23,
                                           planner_inflation_m=.52,
                                           interpretation="A witness exists; no method is given its location"))
    source_files = [p for p in templates.rglob("*") if p.is_file()]
    save(OUT / "scenario_freeze.json", {
        "cases": list(CASES), "seed": 7601,
        "execution_order": [["visible", "nearest"], ["visible", "gain"], ["visible", "aware"],
                            ["moved", "gain"], ["moved", "aware"], ["moved", "nearest"],
                            ["wrong_asset", "aware"], ["wrong_asset", "nearest"], ["wrong_asset", "gain"],
                            ["absent", "nearest"], ["absent", "aware"], ["absent", "gain"]],
        "pipeline_freeze_sha256": sha(OUT / "pipeline_freeze.json"),
        "templates": {str(p.relative_to(REPO)): sha(p) for p in source_files},
        "evaluator_preflight_sha256": sha(oracle / "feasibility.json"),
        "world_geometry_not_method_input": True,
        "outcomes_unopened_at_freeze": True,
    })
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
