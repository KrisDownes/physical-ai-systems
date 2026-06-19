from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"

COLLECT_SCRIPT = APP_DIR / "collect_citibike_snapshots.py"
INGEST_SCRIPT = APP_DIR / "ingest_citibike_snapshots.py"
INTERVAL_SCRIPT = APP_DIR / "build_citibike_reliability_intervals.py"
MAP_SCRIPT = APP_DIR / "build_citibike_reliability_map.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NYC Urban Pulse Citi Bike reliability pipeline."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "citibike",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "urban_pulse.duckdb",
    )

    parser.add_argument(
        "--collect",
        action="store_true",
        help="Fetch new Citi Bike station_status snapshots before rebuilding reports/maps.",
    )

    parser.add_argument(
        "--collect-interval-sec",
        type=int,
        default=65,
        help="Seconds between station_status fetches when --collect is used.",
    )

    parser.add_argument(
        "--collect-iterations",
        type=int,
        default=55,
        help="Number of station_status fetch cycles when --collect is used.",
    )

    parser.add_argument(
        "--force-station-info",
        action="store_true",
        help="When collecting, force refresh of station_information at startup.",
    )

    parser.add_argument(
        "--max-gap-min",
        type=float,
        default=3.0,
        help="Maximum source-update gap allowed when building intervals.",
    )

    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=1.0,
        help="Report window relative to the latest collected interval.",
    )

    parser.add_argument(
        "--min-observed-min",
        type=float,
        default=30.0,
        help="Minimum observed minutes required for a station to appear in reports/maps.",
    )

    parser.add_argument(
        "--window-label",
        type=str,
        default="Latest observed hour",
    )

    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-intervals", action="store_true")
    parser.add_argument("--skip-map", action="store_true")

    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("\n" + "=" * 88)
    print("RUNNING:")
    print(" ".join(command))
    print("=" * 88)

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()

    print("NYC Urban Pulse pipeline")
    print(f"project_root:     {PROJECT_ROOT}")
    print(f"raw_dir:          {args.raw_dir}")
    print(f"db:               {args.db}")
    print(f"collect:          {args.collect}")
    print(f"lookback_hours:   {args.lookback_hours}")
    print(f"min_observed_min: {args.min_observed_min}")

    if args.collect:
        collect_cmd = [
            sys.executable,
            str(COLLECT_SCRIPT),
            "--interval-sec",
            str(args.collect_interval_sec),
            "--iterations",
            str(args.collect_iterations),
            "--raw-dir",
            str(args.raw_dir),
            "--db",
            str(args.db),
            "--continue-on-error",
        ]

        if args.force_station_info:
            collect_cmd.append("--force-station-info")

        run_command(collect_cmd)

    if not args.skip_ingest:
        run_command(
            [
                sys.executable,
                str(INGEST_SCRIPT),
                "--raw-dir",
                str(args.raw_dir),
                "--db",
                str(args.db),
            ]
        )

    if not args.skip_intervals:
        run_command(
            [
                sys.executable,
                str(INTERVAL_SCRIPT),
                "--db",
                str(args.db),
                "--max-gap-min",
                str(args.max_gap_min),
                "--lookback-hours",
                str(args.lookback_hours),
                "--min-observed-min",
                str(args.min_observed_min),
            ]
        )

    if not args.skip_map:
        run_command(
            [
                sys.executable,
                str(MAP_SCRIPT),
                "--min-observed-min",
                str(args.min_observed_min),
                "--window-label",
                args.window_label,
            ]
        )

    print("\nPipeline complete.")
    print("Main outputs:")
    print("  outputs/reports/citibike_interval_summary_by_status.csv")
    print("  outputs/reports/citibike_station_reliability_summary.csv")
    print("  outputs/maps/citibike_station_reliability_map.html")


if __name__ == "__main__":
    main()
