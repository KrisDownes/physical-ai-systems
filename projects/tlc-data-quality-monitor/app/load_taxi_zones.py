from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and load NYC TLC taxi zone lookup table into DuckDB."
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "tlc.duckdb",
        help="DuckDB database path.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "tlc" / "taxi_zone_lookup.csv",
        help="Local output path for zone lookup CSV.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the lookup CSV even if it already exists.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def download_lookup(output_path: Path, force: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"using existing: {output_path}")
        return

    print(f"downloading: {ZONE_LOOKUP_URL}")
    response = requests.get(ZONE_LOOKUP_URL, timeout=60)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    print(f"saved: {output_path}")


def main() -> None:
    args = parse_args()

    db_path = resolve_path(args.db)
    csv_path = resolve_path(args.out)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    download_lookup(csv_path, force=args.force)

    con = duckdb.connect(str(db_path))

    csv_sql = str(csv_path).replace("'", "''")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE dim_taxi_zones AS
        SELECT
            CAST(LocationID AS INTEGER) AS location_id,
            CAST(Borough AS VARCHAR) AS borough,
            CAST(Zone AS VARCHAR) AS zone,
            CAST(service_zone AS VARCHAR) AS service_zone
        FROM read_csv_auto('{csv_sql}', header=true);
        """
    )

    count = con.execute("SELECT COUNT(*) FROM dim_taxi_zones").fetchone()[0]

    print("\nLoaded taxi zones.")
    print(f"db:              {db_path}")
    print(f"csv:             {csv_path}")
    print(f"dim_taxi_zones:  {count} rows")


if __name__ == "__main__":
    main()