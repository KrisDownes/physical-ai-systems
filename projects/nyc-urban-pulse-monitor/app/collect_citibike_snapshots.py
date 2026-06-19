from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"

FETCH_SCRIPT = APP_DIR / "fetch_citibike_snapshot.py"
INGEST_SCRIPT = APP_DIR / "ingest_citibike_snapshots.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Citi Bike station status snapshots repeatedly and ingest them into DuckDB. "
            "Station information is refreshed less often because it is dimension metadata."
        )
    )

    parser.add_argument(
        "--interval-sec",
        type=int,
        default=65,
        help="Seconds between station_status snapshots. 65 avoids duplicate 60-second TTL updates.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of station_status collection cycles.",
    )

    parser.add_argument(
        "--station-info-refresh-hours",
        type=float,
        default=24.0,
        help="Refresh station_information if the latest local snapshot is older than this.",
    )

    parser.add_argument(
        "--force-station-info",
        action="store_true",
        help="Fetch station_information at startup even if a recent snapshot exists.",
    )

    parser.add_argument(
        "--skip-station-info",
        action="store_true",
        help="Do not fetch station_information at startup.",
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
        "--skip-ingest",
        action="store_true",
        help="Fetch raw snapshots but do not ingest them.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue collecting if a fetch or ingest cycle fails.",
    )

    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_str() -> str:
    return utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None

    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def read_observed_at(path: Path) -> datetime | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        return parse_iso_timestamp(doc.get("observed_at_utc"))
    except Exception:
        return None


def latest_station_information_observed_at(raw_dir: Path) -> datetime | None:
    folder = raw_dir / "station_information"

    if not folder.exists():
        return None

    latest: datetime | None = None

    for path in folder.glob("station_information_*.json"):
        observed_at = read_observed_at(path)

        if observed_at is None:
            continue

        if latest is None or observed_at > latest:
            latest = observed_at

    return latest


def station_information_needs_refresh(args: argparse.Namespace) -> bool:
    if args.skip_station_info:
        return False

    if args.force_station_info:
        return True

    latest = latest_station_information_observed_at(args.raw_dir)

    if latest is None:
        return True

    age_hours = (utc_now() - latest).total_seconds() / 3600.0

    return age_hours >= args.station_info_refresh_hours


def run_command(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f"[{utc_now_str()}] Running:")
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def fetch_feed(feed: str, args: argparse.Namespace) -> None:
    run_command(
        [
            sys.executable,
            str(FETCH_SCRIPT),
            "--feed",
            feed,
            "--out-dir",
            str(args.raw_dir),
        ]
    )


def ingest(args: argparse.Namespace) -> None:
    if args.skip_ingest:
        return

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


def run_cycle(args: argparse.Namespace, cycle_number: int) -> None:
    print("\n" + "#" * 80)
    print(f"[{utc_now_str()}] station_status cycle {cycle_number}/{args.iterations}")
    print("#" * 80)

    fetch_feed("station_status", args)
    ingest(args)


def main() -> None:
    args = parse_args()

    if args.interval_sec <= 0:
        raise ValueError("--interval-sec must be positive.")

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive.")

    if not FETCH_SCRIPT.exists():
        raise FileNotFoundError(f"Missing fetch script: {FETCH_SCRIPT}")

    if not INGEST_SCRIPT.exists():
        raise FileNotFoundError(f"Missing ingest script: {INGEST_SCRIPT}")

    print("Citi Bike snapshot collector")
    print(f"project_root:                 {PROJECT_ROOT}")
    print(f"interval_sec:                 {args.interval_sec}")
    print(f"iterations:                   {args.iterations}")
    print(f"station_info_refresh_hours:   {args.station_info_refresh_hours}")
    print(f"force_station_info:           {args.force_station_info}")
    print(f"skip_station_info:            {args.skip_station_info}")
    print(f"raw_dir:                      {args.raw_dir}")
    print(f"db:                           {args.db}")
    print(f"skip_ingest:                  {args.skip_ingest}")
    print(f"continue_on_error:            {args.continue_on_error}")

    if station_information_needs_refresh(args):
        print("\nStation information is missing or stale. Fetching station_information once.")
        try:
            fetch_feed("station_information", args)
            ingest(args)
        except Exception as exc:
            print(f"\n[{utc_now_str()}] ERROR fetching station_information: {exc}")
            if not args.continue_on_error:
                raise
    else:
        latest = latest_station_information_observed_at(args.raw_dir)
        print(f"\nStation information is fresh enough. Latest observed_at_utc: {latest}")

    for cycle_number in range(1, args.iterations + 1):
        cycle_start = time.monotonic()

        try:
            run_cycle(args, cycle_number)
        except Exception as exc:
            print(f"\n[{utc_now_str()}] ERROR during cycle {cycle_number}: {exc}")

            if not args.continue_on_error:
                raise

        if cycle_number == args.iterations:
            break

        elapsed = time.monotonic() - cycle_start
        sleep_sec = max(0.0, args.interval_sec - elapsed)

        print(f"\n[{utc_now_str()}] Sleeping {sleep_sec:.1f} seconds...")
        time.sleep(sleep_sec)

    print("\n" + "=" * 80)
    print(f"[{utc_now_str()}] Collection complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
