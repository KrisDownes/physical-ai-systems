from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"

DOWNLOAD_SCRIPT = APP_DIR / "download_tlc_month.py"
PROFILE_SCRIPT = APP_DIR / "profile_tlc_month.py"
INGEST_SCRIPT = APP_DIR / "ingest_tlc_month.py"
LOAD_ZONES_SCRIPT = APP_DIR / "load_taxi_zones.py"
ANALYZE_SCRIPT = APP_DIR / "analyze_tlc_month.py"
ANALYZE_ZONES_SCRIPT = APP_DIR / "analyze_tlc_zones.py"
VIS_SCRIPT = APP_DIR / "build_pickup_hotspot_map.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full TLC data quality + analytics pipeline."
    )

    parser.add_argument("--taxi-type", default="yellow", choices=["yellow"])
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)

    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "tlc.duckdb",
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
    )
    parser.add_argument(
        "--skip-zones",
        action="store_true",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
    )

    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()

    filename = f"{args.taxi_type}_tripdata_{args.year}-{args.month:02d}.parquet"
    raw_file = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "tlc"
        / args.taxi_type
        / f"{args.year}"
        / f"{args.month:02d}"
        / filename
    )

    print("TLC pipeline")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"taxi_type:    {args.taxi_type}")
    print(f"year:         {args.year}")
    print(f"month:        {args.month:02d}")
    print(f"raw_file:     {raw_file}")
    print(f"db:           {args.db}")

    if not args.skip_download:
        cmd = [
            sys.executable,
            str(DOWNLOAD_SCRIPT),
            "--taxi-type", args.taxi_type,
            "--year", str(args.year),
            "--month", str(args.month),
        ]
        if args.force_download:
            cmd.append("--force")
        run_command(cmd)
    else:
        print("\nSkipping download stage.")

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Expected raw file not found: {raw_file}"
        )

    if not args.skip_profile:
        run_command([
            sys.executable,
            str(PROFILE_SCRIPT),
            "--input", str(raw_file),
        ])
    else:
        print("\nSkipping profile stage.")

    if not args.skip_ingest:
        cmd = [
            sys.executable,
            str(INGEST_SCRIPT),
            "--input", str(raw_file),
            "--taxi-type", args.taxi_type,
            "--year", str(args.year),
            "--month", str(args.month),
            "--db", str(args.db),
        ]
        if args.force_ingest:
            cmd.append("--force")
        run_command(cmd)
    else:
        print("\nSkipping ingest stage.")

    if not args.skip_zones:
        run_command([
            sys.executable,
            str(LOAD_ZONES_SCRIPT),
            "--db", str(args.db),
        ])
    else:
        print("\nSkipping taxi zone load stage.")

    if not args.skip_analysis:
        run_command([
            sys.executable,
            str(ANALYZE_SCRIPT),
            "--db", str(args.db),
            "--taxi-type", args.taxi_type,
            "--year", str(args.year),
            "--month", str(args.month),
        ])

        run_command([
            sys.executable,
            str(ANALYZE_ZONES_SCRIPT),
            "--db", str(args.db),
            "--taxi-type", args.taxi_type,
            "--year", str(args.year),
            "--month", str(args.month),
        ])
    else:
        print("\nSkipping analysis stage.")

    if not args.skip_viz:
        run_command([
            sys.executable,
            str(VIS_SCRIPT),
            "--db", str(args.db),
            "--taxi-type", args.taxi_type,
            "--year", str(args.year),
            "--month", str(args.month),
        ])
    else:
        print("\nSkipping visualization stage.")

    print("\n" + "=" * 80)
    print("Pipeline complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()