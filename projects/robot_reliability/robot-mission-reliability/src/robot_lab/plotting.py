from __future__ import annotations

from pathlib import Path

from .replay import replay_events


def plot_runs(paths: list[Path], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install plotting support with: pip install -e '.[plots]'") from exc

    figure, axis = plt.subplots(figsize=(9, 5.5))
    truth_plotted = False
    for path in paths:
        events = [event for event in replay_events(path) if event["event_type"] == "robot.telemetry"]
        version = events[0]["software_version"]
        truth_x = [event["payload"]["truth"]["x_m"] for event in events]
        truth_y = [event["payload"]["truth"]["y_m"] for event in events]
        est_x = [event["payload"]["estimate"]["x_m"] for event in events]
        est_y = [event["payload"]["estimate"]["y_m"] for event in events]
        axis.plot(truth_x, truth_y, linewidth=2, label=f"actual path ({version})")
        axis.plot(est_x, est_y, linestyle="--", alpha=0.8, label=f"estimated path ({version})")
        if not truth_plotted:
            axis.scatter([2, 4, 6], [0, 0, 0], marker="x", s=80, color="black", label="waypoints")
            truth_plotted = True
    axis.set_title("Mission replay: actual versus believed robot position")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
