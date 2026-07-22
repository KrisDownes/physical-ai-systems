from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import comparison_markdown, evaluate_run, write_report
from .plotting import plot_runs
from .replay import replay_events
from .simulation import RunConfig, run_simulation


def run_experiment(output_dir: Path, make_plot: bool) -> None:
    runs_dir = output_dir / "runs"
    paths: list[Path] = []
    for version in ("v1", "v2"):
        path = runs_dir / f"{version}.jsonl"
        run_id = run_simulation(path, RunConfig(software_version=version))
        paths.append(path)
        print(f"simulated {version}: run_id={run_id} -> {path}")

    metrics = [evaluate_run(path) for path in paths]
    write_report(metrics, output_dir / "report.json")
    comparison = comparison_markdown(metrics)
    (output_dir / "comparison.md").write_text(comparison, encoding="utf-8")
    print("\n" + comparison)
    if make_plot:
        plot_runs(paths, output_dir / "trajectory.png")
        print(f"plot -> {output_dir / 'trajectory.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot mission replay and reliability lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment = subparsers.add_parser("experiment", help="simulate and compare v1 with v2")
    experiment.add_argument("--output", type=Path, default=Path("artifacts"))
    experiment.add_argument("--plot", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="evaluate one run log")
    evaluate.add_argument("path", type=Path)

    replay = subparsers.add_parser("replay", help="print a concise event-time replay")
    replay.add_argument("path", type=Path)

    args = parser.parse_args()
    if args.command == "experiment":
        run_experiment(args.output, args.plot)
    elif args.command == "evaluate":
        print(evaluate_run(args.path))
    elif args.command == "replay":
        for event in replay_events(args.path):
            if event["event_type"] in {"mission.event", "fault.injection"}:
                print(
                    f"t={event['event_time_s']:5.1f}s seq={event['sequence']:04d} "
                    f"{event['event_type']}: {event['payload']}"
                )


if __name__ == "__main__":
    main()
