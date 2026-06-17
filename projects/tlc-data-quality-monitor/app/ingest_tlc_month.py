from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest one NYC TLC yellow taxi Parquet file into DuckDB and build staging/clean tables."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to TLC yellow taxi Parquet file.",
    )

    parser.add_argument(
        "--taxi-type",
        default="yellow",
        choices=["yellow"],
        help="Taxi type. For now this script supports yellow taxi schema.",
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Dataset year, e.g. 2026.",
    )

    parser.add_argument(
        "--month",
        type=int,
        required=True,
        help="Dataset month, 1-12.",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "tlc.duckdb",
        help="DuckDB database path.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and re-ingest this taxi_type/year/month partition if it already exists.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def month_bounds(year: int, month: int) -> tuple[str, str]:
    start = f"{year}-{month:02d}-01"

    if month == 12:
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{month + 1:02d}-01"

    return start, end


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    result = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
        """,
        [table_name],
    ).fetchone()[0]

    return result > 0


def query_one(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None):
    return con.execute(sql, params or []).fetchone()[0]


def main() -> None:
    args = parse_args()

    input_path = resolve_path(args.input)
    db_path = resolve_path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    start_date, end_date = month_bounds(args.year, args.month)

    con = duckdb.connect(str(db_path))

    parquet_path_sql = str(input_path).replace("'", "''")

    print("TLC ingest")
    print(f"input:     {input_path}")
    print(f"db:        {db_path}")
    print(f"taxi_type: {args.taxi_type}")
    print(f"year:      {args.year}")
    print(f"month:     {args.month:02d}")
    print(f"window:    [{start_date}, {end_date})")

    if table_exists(con, "raw_yellow_trips"):
        existing_count = query_one(
            con,
            """
            SELECT COUNT(*)
            FROM raw_yellow_trips
            WHERE taxi_type = ? AND data_year = ? AND data_month = ?
            """,
            [args.taxi_type, args.year, args.month],
        )

        if existing_count > 0 and not args.force:
            raise RuntimeError(
                f"Partition already exists with {existing_count} rows. "
                "Use --force to delete and re-ingest."
            )

        if args.force:
            con.execute(
                """
                DELETE FROM raw_yellow_trips
                WHERE taxi_type = ? AND data_year = ? AND data_month = ?
                """,
                [args.taxi_type, args.year, args.month],
            )

    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS raw_yellow_trips AS
        SELECT
            *,
            CAST(NULL AS VARCHAR) AS taxi_type,
            CAST(NULL AS INTEGER) AS data_year,
            CAST(NULL AS INTEGER) AS data_month,
            CAST(NULL AS VARCHAR) AS source_file,
            CAST(NULL AS TIMESTAMP) AS ingested_at_utc
        FROM read_parquet('{parquet_path_sql}')
        LIMIT 0;
        """
    )

    ingested_at = datetime.now(timezone.utc).replace(tzinfo=None)

    con.execute(
        f"""
        INSERT INTO raw_yellow_trips
        SELECT
            *,
            ? AS taxi_type,
            ? AS data_year,
            ? AS data_month,
            ? AS source_file,
            ? AS ingested_at_utc
        FROM read_parquet('{parquet_path_sql}');
        """,
        [args.taxi_type, args.year, args.month, str(input_path), ingested_at],
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE stg_yellow_trips AS
        SELECT
            taxi_type,
            data_year,
            data_month,
            source_file,
            ingested_at_utc,

            VendorID,
            tpep_pickup_datetime AS pickup_datetime,
            tpep_dropoff_datetime AS dropoff_datetime,
            passenger_count,
            trip_distance,
            RatecodeID,
            store_and_fwd_flag,
            PULocationID AS pickup_location_id,
            DOLocationID AS dropoff_location_id,
            payment_type,
            fare_amount,
            extra,
            mta_tax,
            tip_amount,
            tolls_amount,
            improvement_surcharge,
            total_amount,
            congestion_surcharge,
            Airport_fee,

            CAST(tpep_pickup_datetime AS DATE) AS pickup_date,
            EXTRACT(HOUR FROM tpep_pickup_datetime) AS pickup_hour,

            EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)) / 60.0
                AS duration_min,

            CASE
                WHEN EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)) > 0
                THEN trip_distance / (
                    EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)) / 3600.0
                )
                ELSE NULL
            END AS speed_mph
        FROM raw_yellow_trips;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE clean_yellow_trips AS
        SELECT *
        FROM stg_yellow_trips
        WHERE
            taxi_type = ?
            AND data_year = ?
            AND data_month = ?
            AND pickup_datetime >= CAST(? AS TIMESTAMP)
            AND pickup_datetime < CAST(? AS TIMESTAMP)
            AND dropoff_datetime > pickup_datetime
            AND duration_min > 0
            AND duration_min <= 24 * 60
            AND trip_distance > 0
            AND trip_distance <= 100
            AND speed_mph > 0
            AND speed_mph <= 100
            AND fare_amount >= 0
            AND total_amount >= 0;
        """,
        [args.taxi_type, args.year, args.month, start_date, end_date],
    )

    raw_count = query_one(
        con,
        """
        SELECT COUNT(*)
        FROM raw_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        """,
        [args.taxi_type, args.year, args.month],
    )

    staging_count = query_one(
        con,
        """
        SELECT COUNT(*)
        FROM stg_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        """,
        [args.taxi_type, args.year, args.month],
    )

    clean_count = query_one(con, "SELECT COUNT(*) FROM clean_yellow_trips")

    rejected_count = raw_count - clean_count
    rejected_fraction = rejected_count / raw_count if raw_count else None

    rule_counts = con.execute(
        """
        SELECT
            SUM(CASE WHEN pickup_datetime < CAST(? AS TIMESTAMP)
                       OR pickup_datetime >= CAST(? AS TIMESTAMP)
                     THEN 1 ELSE 0 END) AS outside_target_month,
            SUM(CASE WHEN dropoff_datetime <= pickup_datetime THEN 1 ELSE 0 END) AS invalid_duration_order,
            SUM(CASE WHEN duration_min <= 0 THEN 1 ELSE 0 END) AS non_positive_duration,
            SUM(CASE WHEN duration_min > 24 * 60 THEN 1 ELSE 0 END) AS duration_over_24h,
            SUM(CASE WHEN trip_distance <= 0 THEN 1 ELSE 0 END) AS non_positive_trip_distance,
            SUM(CASE WHEN trip_distance > 100 THEN 1 ELSE 0 END) AS distance_over_100_miles,
            SUM(CASE WHEN speed_mph > 100 THEN 1 ELSE 0 END) AS speed_over_100_mph,
            SUM(CASE WHEN fare_amount < 0 THEN 1 ELSE 0 END) AS negative_fare_amount,
            SUM(CASE WHEN total_amount < 0 THEN 1 ELSE 0 END) AS negative_total_amount
        FROM stg_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        """,
        [start_date, end_date, args.taxi_type, args.year, args.month],
    ).fetchdf().to_dict(orient="records")[0]

    summary = {
        "input_path": str(input_path),
        "db_path": str(db_path),
        "taxi_type": args.taxi_type,
        "year": args.year,
        "month": args.month,
        "target_month_start": start_date,
        "target_month_end_exclusive": end_date,
        "raw_count": raw_count,
        "staging_count": staging_count,
        "clean_count": clean_count,
        "rejected_count": rejected_count,
        "rejected_fraction": rejected_fraction,
        "rule_counts": rule_counts,
    }

    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / f"{args.taxi_type}_{args.year}_{args.month:02d}_ingest_summary.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nIngest complete.")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nsummary: {report_path}")


if __name__ == "__main__":
    main()