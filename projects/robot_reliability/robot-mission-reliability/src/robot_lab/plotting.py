from __future__ import annotations

from pathlib import Path

from .replay import replay_events
from .experiment import ExperimentResult


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

def animate_run(
    result: ExperimentResult,
    output: Path,
) -> None:

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        raise RuntimeError("Install plotting support with: pip install -e '.[plots]'") from exc

    telemetry = [
        event
        for event in replay_events(result.log_path)
        if event["event_type"] == "robot.telemetry"
    ]
    if not telemetry:
        raise ValueError(f"No telemetry events found in {result.log_path}")
    
    times = [event["event_time_s"] for event in telemetry]
    truth_x = [event["payload"]["truth"]["x_m"] for event in telemetry]
    truth_y = [event["payload"]["truth"]["y_m"] for event in telemetry]
    estimated_x = [event["payload"]["estimate"]["x_m"] for event in telemetry]
    estimated_y = [event["payload"]["estimate"]["y_m"] for event in telemetry]
    target_x = [event["payload"]["mission"]["target_x_m"] for event in telemetry]
    target_y = [event["payload"]["mission"]["target_y_m"] for event in telemetry]


    waypoints: list[tuple[float,float]] = []

    for x_m, y_m in zip(target_x, target_y):
        waypoint = (x_m, y_m)
        if waypoint not in waypoints:
            waypoints.append(waypoint)
    
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.set_title(f"Telemetry Replay: {result.scenario_name} Version:{result.software_version} Seed: {result.seed}")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)

    waypoint_x = [x_m for x_m, _ in waypoints]
    waypoint_y = [y_m for _, y_m in waypoints]
    
    all_x = truth_x + estimated_x + waypoint_x
    all_y = truth_y + estimated_y + waypoint_y

    margin_m = 0.5

    axis.set_xlim(
        min(all_x) - margin_m,
        max(all_x) + margin_m,
    )
    axis.set_ylim(
        min(all_y) - margin_m,
        max(all_y) + margin_m,
    )

    axis.scatter(
        waypoint_x,
        waypoint_y,
        color="black",
        marker="x",
        s=80,
        label="Waypoints",
        zorder=3,
    )

    truth_line, = axis.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2,
        label="True path",
    )

    estimated_line, = axis.plot(
        [],
        [],
        color="tab:orange",
        linewidth=2,
        linestyle="--",
        label="Estimated path",
    )

    truth_marker, = axis.plot(
        [],
        [],
        color="tab:blue",
        marker="o",
        markersize=7,
        linestyle="None",
        label="True position",
    )

    estimated_marker, = axis.plot(
        [],
        [],
        color="tab:orange",
        marker="o",
        markersize=7,
        linestyle="None",
        label="Estimated position",
    )

    target_marker, = axis.plot(
        [],
        [],
        color="tab:red",
        marker="*",
        markersize=15,
        linestyle="None",
        label="Active target",
        zorder=4,
    )

    status_text = axis.text(
        0.02,
        0.98,
        "",
        transform=axis.transAxes,
        verticalalignment="top",
    )

    axis.legend(loc="lower right")
    figure.tight_layout()

    def update(frame_index: int):
        path_end = frame_index + 1

        truth_line.set_data(
            truth_x[:path_end],
            truth_y[:path_end],
        )

        estimated_line.set_data(
            estimated_x[:path_end],
            estimated_y[:path_end],
        )

        truth_marker.set_data(
            [truth_x[frame_index]],
            [truth_y[frame_index]],
        )

        estimated_marker.set_data(
            [estimated_x[frame_index]],
            [estimated_y[frame_index]],
        )

        target_marker.set_data(
            [target_x[frame_index]],
            [target_y[frame_index]],
        )

        current_time_s = times[frame_index]

        slip_active = (
            result.scenario.slip_start_s
            <= current_time_s
            < result.scenario.slip_end_s
        )

        slip_status = "ACTIVE" if slip_active else "inactive"

        current_target = (
            target_x[frame_index],
            target_y[frame_index],
        )
        
        target_number = waypoints.index(current_target) + 1

        mission_status = (
            "SUCCESS"
            if result.metrics.mission_success
            else "FAILED"
        )

        status_text.set_text(
            f"Time: {current_time_s:.1f} s\n"
            f"Active target: {target_number}/{len(waypoints)}\n"
            f"Wheel slip: {slip_status}\n"
            f"Mission: {mission_status} "
            f"({result.metrics.waypoints_reached}/{len(waypoints)} waypoints)\n"
            f"Localization RMSE: "
            f"{result.metrics.localization_rmse_m:.3f} m\n"
            f"Cross-track RMSE: "
            f"{result.metrics.cross_track_rmse_m:.3f} m"
        )

        return (
            truth_line,
            estimated_line,
            truth_marker,
            estimated_marker,
            target_marker,
            status_text,
        )
    frame_indices = list(range(0, len(telemetry), 2))

    final_frame_index = len(telemetry) - 1
    if frame_indices[-1] != final_frame_index:
        frame_indices.append(final_frame_index)
    
    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=100,
        blit=True,
        repeat=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        animation.save(
            output,
            writer=PillowWriter(fps=10),
        )
    finally:
        plt.close(figure)