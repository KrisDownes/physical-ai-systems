"""Evaluator-only visual trace of the opened wrong-asset development missions."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from continuous_asset_analysis import truth_pose


def load(path):
    return json.loads(path.read_text())


def draw(ax, path, title):
    pose = truth_pose(path)
    handoff = load(path / "handoff.json")["estimated_state"]["time"]
    before = pose[:, 0] <= handoff
    after = pose[:, 0] >= handoff
    ax.plot(pose[before, 1], pose[before, 2], color="#727272", lw=1.4, label="acquisition (judge trace)")
    ax.plot(pose[after, 1], pose[after, 2], color="#1769aa", lw=1.8, label="inspection (judge trace)")
    ax.scatter(*pose[0, 1:3], c="#555555", marker="s", s=45, label="initial dock")
    ax.scatter(*pose[-1, 1:3], c="#1769aa", marker="X", s=70, label="termination")
    scene = load(path / "private/scene.json")
    for asset in scene["assets"]:
        marker = "*" if asset["id"] == 7 else "D"
        color = "#b6272b" if asset["id"] == 7 else "#a36200"
        ax.scatter(*asset["pose"][:2], marker=marker, s=110, color=color,
                   label=f"ID{asset['id']} (judge only)")
    finish = next(json.loads(line) for line in (path / "inspection/method.jsonl").read_text().splitlines()
                  if json.loads(line).get("event") == "finish")
    for i, station in enumerate(finish["visited_search_positions"]):
        ax.scatter(*station, color="#0c8053", marker="o", s=25)
        ax.annotate(str(i + 1), station, xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set(title=title, xlim=(-3.3, 3.3), ylim=(-3.3, 3.3), xlabel="site x (m)", ylabel="site y (m)")
    ax.set_aspect("equal"); ax.grid(alpha=.25)


def main():
    base = ROOT / "results/post_repair_connected_20261027/mission_wrong_asset"
    gain = ROOT / "results/visibility_search_20261027/mission_wrong_asset"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharex=True, sharey=True)
    draw(axes[0], base, "Nearest: late ID7, approach budget exhausted")
    draw(axes[1], gain, "Visibility gain: valid ID7 inspection")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=.20, wspace=.18, top=.90)
    output = ROOT / "results/visibility_search_20261027/wrong_asset_routes.png"
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
