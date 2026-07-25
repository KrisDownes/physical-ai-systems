from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import comparison_markdown, evaluate_run, write_report
from .plotting import plot_runs
from .replay import replay_events
from .simulation import RunConfig, run_simulation
from .experiment import run_experiment_matrix


def run_experiment(output_dir: Path) -> None:
    results = run_experiment_matrix(output_dir)

    for result in results:
        print(
            f"scenario={result.scenario_name} "
            f"version={result.software_version} "
            f"seed={result.seed} "
            f"success={result.metrics.mission_success} "
            f"waypoints={result.metrics.waypoints_reached} "
            f"duration_s={result.metrics.duration_s:.1f} "
            f"log={result.log_path}"
        )

    print(f"completed {len(results)} runs")



def main() -> None:
    parser = argparse.ArgumentParser(description="Robot mission replay and reliability lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment = subparsers.add_parser("experiment", help="simulate and compare v1 with v2")
    experiment.add_argument("--output", type=Path, default=Path("artifacts"))

    evaluate = subparsers.add_parser("evaluate", help="evaluate one run log")
    evaluate.add_argument("path", type=Path)

    replay = subparsers.add_parser("replay", help="print a concise event-time replay")
    replay.add_argument("path", type=Path)

    args = parser.parse_args()
    if args.command == "experiment":
        run_experiment(args.output)
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
