from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest one TLC month into persistent raw/staging/clean DuckDB layers.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--taxi-type", default="yellow", choices=["yellow"])
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "db" / "tlc.duckdb")
    p.add_argument("--force", action="store_true", help="Reload this raw partition before rebuilding the clean partition.")
    return p.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def month_bounds(year: int, month: int) -> tuple[str, str]:
    start = f"{year}-{month:02d}-01"
    end = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"
    return start, end


def q1(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None):
    return con.execute(sql, params or []).fetchone()[0]


def ensure_raw_table(con: duckdb.DuckDBPyConnection, parquet_path_sql: str) -> None:
    con.execute(f"""
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
    """)


def rebuild_staging_view(con: duckdb.DuckDBPyConnection) -> None:
    # In old versions of this project, stg_yellow_trips may have been a TABLE.
    # In the new design, it should be a VIEW over all raw partitions.
    existing = con.execute(
        """
        SELECT table_type
        FROM information_schema.tables
        WHERE table_name = 'stg_yellow_trips'
        """
    ).fetchall()

    if existing:
        table_type = existing[0][0]
        if table_type == "VIEW":
            con.execute("DROP VIEW stg_yellow_trips;")
        else:
            con.execute("DROP TABLE stg_yellow_trips;")

    con.execute(
        """
        CREATE VIEW stg_yellow_trips AS
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


def ensure_clean_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS clean_yellow_trips AS
        SELECT *
        FROM stg_yellow_trips
        WHERE 1 = 0;
    """)


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    db_path = resolve_path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    start_date, end_date = month_bounds(args.year, args.month)
    parquet_path_sql = str(input_path).replace("'", "''")
    con = duckdb.connect(str(db_path))

    print("TLC ingest")
    print(f"input:     {input_path}")
    print(f"db:        {db_path}")
    print(f"taxi_type: {args.taxi_type}")
    print(f"year:      {args.year}")
    print(f"month:     {args.month:02d}")
    print(f"window:    [{start_date}, {end_date})")

    ensure_raw_table(con, parquet_path_sql)

    existing_raw = q1(con, """
        SELECT COUNT(*) FROM raw_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
    """, [args.taxi_type, args.year, args.month])

    if existing_raw and args.force:
        print(f"Deleting existing raw partition rows: {existing_raw}")
        con.execute("""
            DELETE FROM raw_yellow_trips
            WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        """, [args.taxi_type, args.year, args.month])
        existing_raw = 0

    if existing_raw:
        print(f"Raw partition already loaded; skipping raw insert. rows={existing_raw}")
    else:
        ingested_at = datetime.now(timezone.utc).replace(tzinfo=None)
        con.execute(f"""
            INSERT INTO raw_yellow_trips
            SELECT *, ? AS taxi_type, ? AS data_year, ? AS data_month, ? AS source_file, ? AS ingested_at_utc
            FROM read_parquet('{parquet_path_sql}');
        """, [args.taxi_type, args.year, args.month, str(input_path), ingested_at])

    rebuild_staging_view(con)
    ensure_clean_table(con)

    existing_clean = q1(con, """
        SELECT COUNT(*) FROM clean_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
    """, [args.taxi_type, args.year, args.month])

    if existing_clean:
        print(f"Replacing existing clean partition rows: {existing_clean}")
        con.execute("""
            DELETE FROM clean_yellow_trips
            WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        """, [args.taxi_type, args.year, args.month])

    con.execute("""
        INSERT INTO clean_yellow_trips
        SELECT *
        FROM stg_yellow_trips
        WHERE taxi_type = ?
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
    """, [args.taxi_type, args.year, args.month, start_date, end_date])

    raw_count = q1(con, """
        SELECT COUNT(*) FROM raw_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
    """, [args.taxi_type, args.year, args.month])
    clean_count = q1(con, """
        SELECT COUNT(*) FROM clean_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
    """, [args.taxi_type, args.year, args.month])
    clean_total = q1(con, "SELECT COUNT(*) FROM clean_yellow_trips")

    rule_counts = con.execute("""
        SELECT
            SUM(CASE WHEN pickup_datetime < CAST(? AS TIMESTAMP) OR pickup_datetime >= CAST(? AS TIMESTAMP) THEN 1 ELSE 0 END) AS outside_target_month,
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
    """, [start_date, end_date, args.taxi_type, args.year, args.month]).fetchdf().to_dict(orient="records")[0]

    clean_partitions = con.execute("""
        SELECT data_year, data_month, COUNT(*) AS clean_count
        FROM clean_yellow_trips
        WHERE taxi_type = ?
        GROUP BY data_year, data_month
        ORDER BY data_year, data_month
    """, [args.taxi_type]).fetchdf().to_dict(orient="records")

    summary = {
        "input_path": str(input_path),
        "db_path": str(db_path),
        "taxi_type": args.taxi_type,
        "year": args.year,
        "month": args.month,
        "raw_count": raw_count,
        "clean_count": clean_count,
        "clean_total_count_all_loaded_partitions": clean_total,
        "rejected_count": raw_count - clean_count,
        "rejected_fraction": (raw_count - clean_count) / raw_count if raw_count else None,
        "rule_counts": rule_counts,
        "loaded_clean_partitions": clean_partitions,
    }

    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.taxi_type}_{args.year}_{args.month:02d}_ingest_summary.json"
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\nIngest complete.")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nsummary: {report_path}")


if __name__ == "__main__":
    main()
