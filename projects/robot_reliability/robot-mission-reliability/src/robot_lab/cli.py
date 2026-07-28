from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import evaluate_run
from .experiment import run_experiment_matrix, write_results_csv
from .plotting import (
    animate_run,
    plot_cross_track_rmse,
    plot_success_rates,
)
from .replay import replay_events
from .reporting import (
    summarize_results,
    write_scenario_comparison,
)


def run_experiment(output_dir: Path, docs_dir: Path,) -> None:
    results = run_experiment_matrix(output_dir)
    results_path = output_dir / "results.csv"
    write_results_csv(results, results_path)

    summaries = summarize_results(results)

    report_path = docs_dir / "scenario_comparison.md"
    success_plot_path = docs_dir / "assets" / "success_rate.svg"
    cross_track_plot_path = docs_dir / "assets" / "cross_track_rmse.svg"
    replay_path = docs_dir / "assets" / "severe_slip_v2_seed_0.gif"

    write_scenario_comparison(summaries, report_path)
    plot_success_rates(summaries, success_plot_path)
    plot_cross_track_rmse(summaries, cross_track_plot_path)

    replay_result = next(
        (
            result
            for result in results
            if result.scenario_name == "severe_slip"
            and result.software_version == "v2"
            and result.seed == 0
        ),
        None,
    )

    if replay_result is None:
        raise ValueError("severe_slip/v2/seed 0 run was not found")

    animate_run(
        result=replay_result,
        output=replay_path,
    )

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

    print(f"results={results_path}")
    print(f"comparison report={report_path}")
    print(f"success plot={success_plot_path}")
    print(f"cross-track plot={cross_track_plot_path}")
    print(f"replay={replay_path}")
    print(f"completed {len(results)} runs")



def main() -> None:
    parser = argparse.ArgumentParser(description="Robot mission replay and reliability lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment = subparsers.add_parser("experiment", help="simulate and compare v1 with v2")
    experiment.add_argument("--output", type=Path, default=Path("artifacts"))
    experiment.add_argument(
        "--docs-output",
        type=Path,
        default=Path("docs"),
    )

    evaluate = subparsers.add_parser("evaluate", help="evaluate one run log")
    evaluate.add_argument("path", type=Path)

    replay = subparsers.add_parser("replay", help="print a concise event-time replay")
    replay.add_argument("path", type=Path)

    args = parser.parse_args()
    if args.command == "experiment":
        run_experiment(args.output, args.docs_output)
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
