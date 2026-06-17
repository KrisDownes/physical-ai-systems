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
ANALYZE_PERIOD_SCRIPT = APP_DIR / "analyze_tlc_period.py"
MONTHLY_VIZ_SCRIPT = APP_DIR / "build_pickup_density_hybrid_taxi_brand.py"
PERIOD_VIZ_SCRIPT = APP_DIR / "build_pickup_density_period_hybrid_taxi_brand.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the TLC data quality + analytics + visualization pipeline.")
    p.add_argument("--taxi-type", default="yellow", choices=["yellow"])
    p.add_argument("--year", type=int, required=True)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--month", type=int, help="Single month to process, 1-12.")
    group.add_argument("--months", type=str, help="Comma-separated months, e.g. 1,2,3.")
    group.add_argument("--month-range", type=str, help="Inclusive range, e.g. 1-6.")

    p.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "db" / "tlc.duckdb")
    p.add_argument("--zones-geometry", type=Path, default=PROJECT_ROOT / "data" / "raw" / "tlc" / "taxi_zones_4326.parquet")

    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-profile", action="store_true")
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument("--skip-zones", action="store_true")
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--skip-viz", action="store_true")
    p.add_argument("--skip-monthly-viz", action="store_true")
    p.add_argument("--skip-period-reports", action="store_true")
    p.add_argument("--skip-period-viz", action="store_true")

    p.add_argument("--force-download", action="store_true")
    p.add_argument("--force-ingest", action="store_true")

    p.add_argument("--max-elevation", type=float, default=1800.0)
    p.add_argument("--start-hour", type=int, default=6)
    p.add_argument("--ms-per-hour", type=int, default=650)
    return p.parse_args()


def parse_months(args: argparse.Namespace) -> list[int]:
    if args.month is not None:
        months = [args.month]
    elif args.months is not None:
        months = [int(m.strip()) for m in args.months.split(",") if m.strip()]
    else:
        start_s, end_s = args.month_range.split("-", maxsplit=1)
        start, end = int(start_s), int(end_s)
        if end < start:
            raise ValueError("--month-range end must be >= start")
        months = list(range(start, end + 1))

    bad = [m for m in months if m < 1 or m > 12]
    if bad:
        raise ValueError(f"Invalid months: {bad}; expected 1-12")

    seen = set()
    out = []
    for m in months:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def check_script(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing script: {path}")


def raw_file_path(taxi_type: str, year: int, month: int) -> Path:
    filename = f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"
    return PROJECT_ROOT / "data" / "raw" / "tlc" / taxi_type / f"{year}" / f"{month:02d}" / filename


def run_command(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(command))
    print("=" * 80)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_month(args: argparse.Namespace, month: int, zones_loaded: bool) -> bool:
    raw_file = raw_file_path(args.taxi_type, args.year, month)

    print("\n" + "#" * 80)
    print(f"Processing {args.taxi_type} {args.year}-{month:02d}")
    print(f"raw_file: {raw_file}")
    print("#" * 80)

    if not args.skip_download:
        cmd = [sys.executable, str(DOWNLOAD_SCRIPT), "--taxi-type", args.taxi_type, "--year", str(args.year), "--month", str(month)]
        if args.force_download:
            cmd.append("--force")
        run_command(cmd)
    else:
        print("\nSkipping download stage.")

    if not raw_file.exists():
        raise FileNotFoundError(f"Expected raw file not found: {raw_file}")

    if not args.skip_profile:
        run_command([sys.executable, str(PROFILE_SCRIPT), "--input", str(raw_file)])
    else:
        print("\nSkipping profile stage.")

    if not args.skip_ingest:
        cmd = [
            sys.executable, str(INGEST_SCRIPT),
            "--input", str(raw_file),
            "--taxi-type", args.taxi_type,
            "--year", str(args.year),
            "--month", str(month),
            "--db", str(args.db),
        ]
        if args.force_ingest:
            cmd.append("--force")
        run_command(cmd)
    else:
        print("\nSkipping ingest stage.")

    if not args.skip_zones and not zones_loaded:
        run_command([sys.executable, str(LOAD_ZONES_SCRIPT), "--db", str(args.db)])
        zones_loaded = True
    elif args.skip_zones:
        print("\nSkipping taxi zone load stage.")

    if not args.skip_analysis:
        run_command([sys.executable, str(ANALYZE_SCRIPT), "--db", str(args.db), "--taxi-type", args.taxi_type, "--year", str(args.year), "--month", str(month)])
        run_command([sys.executable, str(ANALYZE_ZONES_SCRIPT), "--db", str(args.db), "--taxi-type", args.taxi_type, "--year", str(args.year), "--month", str(month)])
    else:
        print("\nSkipping analysis stage.")

    if not args.skip_viz and not args.skip_monthly_viz:
        zones_geometry = resolve_path(args.zones_geometry)
        if not zones_geometry.exists():
            raise FileNotFoundError(f"Taxi zone geometry file not found: {zones_geometry}")
        run_command([
            sys.executable, str(MONTHLY_VIZ_SCRIPT),
            "--db", str(args.db),
            "--taxi-type", args.taxi_type,
            "--year", str(args.year),
            "--month", str(month),
            "--zones", str(zones_geometry),
            "--max-elevation", str(args.max_elevation),
            "--start-hour", str(args.start_hour),
            "--ms-per-hour", str(args.ms_per_hour),
        ])
    else:
        print("\nSkipping monthly visualization stage.")

    return zones_loaded


def run_period_outputs(args: argparse.Namespace, months: list[int]) -> None:
    if len(months) <= 1:
        return
    start_month, end_month = min(months), max(months)

    if not args.skip_analysis and not args.skip_period_reports:
        run_command([sys.executable, str(ANALYZE_PERIOD_SCRIPT), "--db", str(args.db), "--taxi-type", args.taxi_type, "--year", str(args.year), "--start-month", str(start_month), "--end-month", str(end_month)])
    else:
        print("\nSkipping period report stage.")

    if not args.skip_viz and not args.skip_period_viz:
        zones_geometry = resolve_path(args.zones_geometry)
        if not zones_geometry.exists():
            raise FileNotFoundError(f"Taxi zone geometry file not found: {zones_geometry}")
        run_command([
            sys.executable, str(PERIOD_VIZ_SCRIPT),
            "--db", str(args.db),
            "--taxi-type", args.taxi_type,
            "--year", str(args.year),
            "--start-month", str(start_month),
            "--end-month", str(end_month),
            "--zones", str(zones_geometry),
            "--max-elevation", str(args.max_elevation),
            "--start-hour", str(args.start_hour),
            "--ms-per-hour", str(args.ms_per_hour),
        ])
    else:
        print("\nSkipping period visualization stage.")


def main() -> None:
    args = parse_args()
    args.db = resolve_path(args.db)
    months = parse_months(args)

    for script in [DOWNLOAD_SCRIPT, PROFILE_SCRIPT, INGEST_SCRIPT, LOAD_ZONES_SCRIPT, ANALYZE_SCRIPT, ANALYZE_ZONES_SCRIPT, ANALYZE_PERIOD_SCRIPT, MONTHLY_VIZ_SCRIPT, PERIOD_VIZ_SCRIPT]:
        check_script(script)

    print("TLC pipeline")
    print(f"project_root:   {PROJECT_ROOT}")
    print(f"taxi_type:      {args.taxi_type}")
    print(f"year:           {args.year}")
    print(f"months:         {', '.join(f'{m:02d}' for m in months)}")
    print(f"db:             {args.db}")

    zones_loaded = False
    completed = []
    for month in months:
        zones_loaded = run_month(args, month, zones_loaded)
        completed.append(month)

    run_period_outputs(args, completed)

    print("\n" + "=" * 80)
    print("Pipeline complete.")
    print(f"Completed months: {', '.join(f'{m:02d}' for m in completed)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
