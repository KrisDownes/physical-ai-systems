from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile one NYC TLC monthly Parquet file with DuckDB."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to TLC Parquet file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def query_one(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def main() -> None:
    args = parse_args()

    input_path = resolve_path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output is None:
        output_path = (
            PROJECT_ROOT
            / "outputs"
            / "reports"
            / f"{input_path.stem}_profile.json"
        )
    else:
        output_path = resolve_path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")

    parquet_path_sql = str(input_path).replace("'", "''")

    con.execute(
        f"""
        CREATE VIEW trips AS
        SELECT *
        FROM read_parquet('{parquet_path_sql}');
        """
    )

    columns = con.execute("DESCRIBE trips").fetchdf().to_dict(orient="records")

    report = {
        "profiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "file_size_mb": input_path.stat().st_size / 1_000_000,
        "schema": columns,
        "row_count": query_one(con, "SELECT COUNT(*) FROM trips"),
        "date_range": con.execute(
            """
            SELECT
                MIN(tpep_pickup_datetime) AS min_pickup_datetime,
                MAX(tpep_pickup_datetime) AS max_pickup_datetime,
                MIN(tpep_dropoff_datetime) AS min_dropoff_datetime,
                MAX(tpep_dropoff_datetime) AS max_dropoff_datetime
            FROM trips
            """
        ).fetchdf().to_dict(orient="records")[0],
        "null_counts": con.execute(
            """
            SELECT
                SUM(CASE WHEN tpep_pickup_datetime IS NULL THEN 1 ELSE 0 END) AS null_pickup_datetime,
                SUM(CASE WHEN tpep_dropoff_datetime IS NULL THEN 1 ELSE 0 END) AS null_dropoff_datetime,
                SUM(CASE WHEN PULocationID IS NULL THEN 1 ELSE 0 END) AS null_pu_location,
                SUM(CASE WHEN DOLocationID IS NULL THEN 1 ELSE 0 END) AS null_do_location,
                SUM(CASE WHEN trip_distance IS NULL THEN 1 ELSE 0 END) AS null_trip_distance,
                SUM(CASE WHEN fare_amount IS NULL THEN 1 ELSE 0 END) AS null_fare_amount,
                SUM(CASE WHEN total_amount IS NULL THEN 1 ELSE 0 END) AS null_total_amount
            FROM trips
            """
        ).fetchdf().to_dict(orient="records")[0],
        "quality_checks": con.execute(
            """
            WITH base AS (
                SELECT
                    *,
                    EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)) / 60.0 AS duration_min,
                    CASE
                        WHEN EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)) > 0
                        THEN trip_distance / (EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)) / 3600.0)
                        ELSE NULL
                    END AS speed_mph
                FROM trips
            )
            SELECT
                SUM(CASE WHEN tpep_dropoff_datetime < tpep_pickup_datetime THEN 1 ELSE 0 END) AS pickup_after_dropoff,
                SUM(CASE WHEN trip_distance <= 0 THEN 1 ELSE 0 END) AS non_positive_trip_distance,
                SUM(CASE WHEN fare_amount < 0 THEN 1 ELSE 0 END) AS negative_fare_amount,
                SUM(CASE WHEN total_amount < 0 THEN 1 ELSE 0 END) AS negative_total_amount,
                SUM(CASE WHEN duration_min <= 0 THEN 1 ELSE 0 END) AS non_positive_duration,
                SUM(CASE WHEN duration_min > 24 * 60 THEN 1 ELSE 0 END) AS duration_over_24h,
                SUM(CASE WHEN speed_mph > 100 THEN 1 ELSE 0 END) AS speed_over_100_mph,
                SUM(CASE WHEN passenger_count < 0 THEN 1 ELSE 0 END) AS negative_passenger_count
            FROM base
            """
        ).fetchdf().to_dict(orient="records")[0],
        "summary_stats": con.execute(
            """
            WITH base AS (
                SELECT
                    *,
                    EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)) / 60.0 AS duration_min,
                    CASE
                        WHEN EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)) > 0
                        THEN trip_distance / (EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)) / 3600.0)
                        ELSE NULL
                    END AS speed_mph
                FROM trips
                WHERE
                    tpep_dropoff_datetime > tpep_pickup_datetime
                    AND trip_distance > 0
            )
            SELECT
                AVG(trip_distance) AS avg_trip_distance_miles,
                MEDIAN(trip_distance) AS median_trip_distance_miles,
                MAX(trip_distance) AS max_trip_distance_miles,
                AVG(duration_min) AS avg_duration_min,
                MEDIAN(duration_min) AS median_duration_min,
                AVG(speed_mph) AS avg_speed_mph,
                MEDIAN(speed_mph) AS median_speed_mph,
                MAX(speed_mph) AS max_speed_mph,
                AVG(fare_amount) AS avg_fare_amount,
                MEDIAN(fare_amount) AS median_fare_amount,
                AVG(total_amount) AS avg_total_amount,
                MEDIAN(total_amount) AS median_total_amount
            FROM base
            """
        ).fetchdf().to_dict(orient="records")[0],
        "payment_type_counts": con.execute(
            """
            SELECT
                payment_type,
                COUNT(*) AS trip_count
            FROM trips
            GROUP BY payment_type
            ORDER BY trip_count DESC
            """
        ).fetchdf().to_dict(orient="records"),
        "rate_code_counts": con.execute(
            """
            SELECT
                RatecodeID,
                COUNT(*) AS trip_count
            FROM trips
            GROUP BY RatecodeID
            ORDER BY trip_count DESC
            """
        ).fetchdf().to_dict(orient="records"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("TLC monthly profile")
    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print()
    print(json.dumps(
        {
            "row_count": report["row_count"],
            "date_range": report["date_range"],
            "quality_checks": report["quality_checks"],
            "summary_stats": report["summary_stats"],
        },
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()