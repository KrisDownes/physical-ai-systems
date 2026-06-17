from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate analysis reports from clean/staging NYC TLC DuckDB tables."
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "tlc.duckdb",
        help="DuckDB database path.",
    )

    parser.add_argument(
        "--taxi-type",
        default="yellow",
        choices=["yellow"],
        help="Taxi type.",
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Dataset year.",
    )

    parser.add_argument(
        "--month",
        type=int,
        required=True,
        help="Dataset month.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports",
        help="Output report directory.",
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


def save_csv(con: duckdb.DuckDBPyConnection, sql: str, path: Path, params: list | None = None) -> None:
    df = con.execute(sql, params or []).fetchdf()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved: {path} rows={len(df)}")


def fetch_one_record(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> dict:
    df = con.execute(sql, params or []).fetchdf()
    if len(df) == 0:
        return {}
    return df.to_dict(orient="records")[0]


def main() -> None:
    args = parse_args()

    db_path = resolve_path(args.db)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    start_date, end_date = month_bounds(args.year, args.month)
    prefix = f"{args.taxi_type}_{args.year}_{args.month:02d}"

    con = duckdb.connect(str(db_path))

    print("TLC analysis")
    print(f"db:        {db_path}")
    print(f"taxi_type: {args.taxi_type}")
    print(f"year:      {args.year}")
    print(f"month:     {args.month:02d}")
    print(f"out_dir:   {out_dir}")

    monthly_summary = fetch_one_record(
        con,
        """
        SELECT
            COUNT(*) AS clean_trip_count,
            MIN(pickup_datetime) AS min_pickup_datetime,
            MAX(pickup_datetime) AS max_pickup_datetime,

            SUM(total_amount) AS gross_total_amount,
            SUM(fare_amount) AS gross_fare_amount,
            SUM(tip_amount) AS gross_tip_amount,
            SUM(tolls_amount) AS gross_tolls_amount,

            AVG(total_amount) AS avg_total_amount,
            MEDIAN(total_amount) AS median_total_amount,

            AVG(fare_amount) AS avg_fare_amount,
            MEDIAN(fare_amount) AS median_fare_amount,

            AVG(tip_amount) AS avg_tip_amount,
            MEDIAN(tip_amount) AS median_tip_amount,

            AVG(trip_distance) AS avg_trip_distance_miles,
            MEDIAN(trip_distance) AS median_trip_distance_miles,
            MAX(trip_distance) AS max_trip_distance_miles,

            AVG(duration_min) AS avg_duration_min,
            MEDIAN(duration_min) AS median_duration_min,
            MAX(duration_min) AS max_duration_min,

            AVG(speed_mph) AS avg_speed_mph,
            MEDIAN(speed_mph) AS median_speed_mph,
            MAX(speed_mph) AS max_speed_mph
        FROM clean_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        """,
        [args.taxi_type, args.year, args.month],
    )

    raw_vs_clean_summary = fetch_one_record(
        con,
        """
        SELECT
            raw.raw_count,
            clean.clean_count,
            raw.raw_count - clean.clean_count AS rejected_count,
            (raw.raw_count - clean.clean_count)::DOUBLE / raw.raw_count AS rejected_fraction
        FROM
            (
                SELECT COUNT(*) AS raw_count
                FROM raw_yellow_trips
                WHERE taxi_type = ? AND data_year = ? AND data_month = ?
            ) raw,
            (
                SELECT COUNT(*) AS clean_count
                FROM clean_yellow_trips
                WHERE taxi_type = ? AND data_year = ? AND data_month = ?
            ) clean
        """,
        [args.taxi_type, args.year, args.month, args.taxi_type, args.year, args.month],
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "taxi_type": args.taxi_type,
        "year": args.year,
        "month": args.month,
        "target_month_start": start_date,
        "target_month_end_exclusive": end_date,
        "raw_vs_clean": raw_vs_clean_summary,
        "monthly_summary": monthly_summary,
    }

    summary_path = out_dir / f"{prefix}_monthly_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"saved: {summary_path}")

    save_csv(
        con,
        """
        SELECT
            pickup_date,
            COUNT(*) AS trip_count,
            SUM(total_amount) AS gross_total_amount,
            SUM(fare_amount) AS gross_fare_amount,
            SUM(tip_amount) AS gross_tip_amount,
            AVG(total_amount) AS avg_total_amount,
            AVG(trip_distance) AS avg_trip_distance_miles,
            AVG(duration_min) AS avg_duration_min,
            AVG(speed_mph) AS avg_speed_mph
        FROM clean_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        GROUP BY pickup_date
        ORDER BY pickup_date
        """,
        out_dir / f"{prefix}_daily_revenue.csv",
        [args.taxi_type, args.year, args.month],
    )

    save_csv(
        con,
        """
        SELECT
            pickup_hour,
            COUNT(*) AS trip_count,
            SUM(total_amount) AS gross_total_amount,
            AVG(total_amount) AS avg_total_amount,
            AVG(trip_distance) AS avg_trip_distance_miles,
            AVG(duration_min) AS avg_duration_min,
            AVG(speed_mph) AS avg_speed_mph
        FROM clean_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        GROUP BY pickup_hour
        ORDER BY pickup_hour
        """,
        out_dir / f"{prefix}_hourly_demand.csv",
        [args.taxi_type, args.year, args.month],
    )

    save_csv(
        con,
        """
        SELECT
            CAST(STRFTIME(pickup_datetime, '%w') AS INTEGER) AS day_of_week,
            pickup_hour,
            COUNT(*) AS trip_count,
            SUM(total_amount) AS gross_total_amount,
            AVG(total_amount) AS avg_total_amount,
            AVG(trip_distance) AS avg_trip_distance_miles,
            AVG(duration_min) AS avg_duration_min
        FROM clean_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        GROUP BY day_of_week, pickup_hour
        ORDER BY day_of_week, pickup_hour
        """,
        out_dir / f"{prefix}_weekday_hour_demand.csv",
        [args.taxi_type, args.year, args.month],
    )

    save_csv(
        con,
        """
        SELECT
            pickup_location_id,
            COUNT(*) AS trip_count,
            SUM(total_amount) AS gross_total_amount,
            AVG(total_amount) AS avg_total_amount,
            AVG(trip_distance) AS avg_trip_distance_miles,
            AVG(duration_min) AS avg_duration_min
        FROM clean_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        GROUP BY pickup_location_id
        ORDER BY trip_count DESC
        LIMIT 25
        """,
        out_dir / f"{prefix}_top_pickup_locations.csv",
        [args.taxi_type, args.year, args.month],
    )

    save_csv(
        con,
        """
        SELECT
            pickup_location_id,
            dropoff_location_id,
            COUNT(*) AS trip_count,
            SUM(total_amount) AS gross_total_amount,
            AVG(total_amount) AS avg_total_amount,
            AVG(trip_distance) AS avg_trip_distance_miles,
            AVG(duration_min) AS avg_duration_min
        FROM clean_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        GROUP BY pickup_location_id, dropoff_location_id
        ORDER BY trip_count DESC
        LIMIT 50
        """,
        out_dir / f"{prefix}_top_routes.csv",
        [args.taxi_type, args.year, args.month],
    )

    save_csv(
        con,
        """
        WITH anomaly_flags AS (
            SELECT
                taxi_type,
                data_year,
                data_month,
                pickup_datetime,
                dropoff_datetime,
                pickup_location_id,
                dropoff_location_id,
                passenger_count,
                trip_distance,
                duration_min,
                speed_mph,
                fare_amount,
                total_amount,

                pickup_datetime < CAST(? AS TIMESTAMP)
                    OR pickup_datetime >= CAST(? AS TIMESTAMP)
                    AS outside_target_month,

                dropoff_datetime <= pickup_datetime AS invalid_duration_order,
                duration_min <= 0 AS non_positive_duration,
                duration_min > 24 * 60 AS duration_over_24h,
                trip_distance <= 0 AS non_positive_trip_distance,
                trip_distance > 100 AS distance_over_100_miles,
                speed_mph > 100 AS speed_over_100_mph,
                fare_amount < 0 AS negative_fare_amount,
                total_amount < 0 AS negative_total_amount
            FROM stg_yellow_trips
            WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        )
        SELECT *
        FROM anomaly_flags
        WHERE
            outside_target_month
            OR invalid_duration_order
            OR non_positive_duration
            OR duration_over_24h
            OR non_positive_trip_distance
            OR distance_over_100_miles
            OR speed_over_100_mph
            OR negative_fare_amount
            OR negative_total_amount
        ORDER BY
            speed_mph DESC NULLS LAST,
            trip_distance DESC NULLS LAST,
            ABS(total_amount) DESC NULLS LAST
        LIMIT 500
        """,
        out_dir / f"{prefix}_anomaly_samples.csv",
        [start_date, end_date, args.taxi_type, args.year, args.month],
    )

    save_csv(
        con,
        """
        WITH anomaly_flags AS (
            SELECT
                pickup_datetime < CAST(? AS TIMESTAMP)
                    OR pickup_datetime >= CAST(? AS TIMESTAMP)
                    AS outside_target_month,

                dropoff_datetime <= pickup_datetime AS invalid_duration_order,
                duration_min <= 0 AS non_positive_duration,
                duration_min > 24 * 60 AS duration_over_24h,
                trip_distance <= 0 AS non_positive_trip_distance,
                trip_distance > 100 AS distance_over_100_miles,
                speed_mph > 100 AS speed_over_100_mph,
                fare_amount < 0 AS negative_fare_amount,
                total_amount < 0 AS negative_total_amount
            FROM stg_yellow_trips
            WHERE taxi_type = ? AND data_year = ? AND data_month = ?
        )
        SELECT 'outside_target_month' AS anomaly_type, SUM(CASE WHEN outside_target_month THEN 1 ELSE 0 END) AS row_count FROM anomaly_flags
        UNION ALL
        SELECT 'invalid_duration_order', SUM(CASE WHEN invalid_duration_order THEN 1 ELSE 0 END) FROM anomaly_flags
        UNION ALL
        SELECT 'non_positive_duration', SUM(CASE WHEN non_positive_duration THEN 1 ELSE 0 END) FROM anomaly_flags
        UNION ALL
        SELECT 'duration_over_24h', SUM(CASE WHEN duration_over_24h THEN 1 ELSE 0 END) FROM anomaly_flags
        UNION ALL
        SELECT 'non_positive_trip_distance', SUM(CASE WHEN non_positive_trip_distance THEN 1 ELSE 0 END) FROM anomaly_flags
        UNION ALL
        SELECT 'distance_over_100_miles', SUM(CASE WHEN distance_over_100_miles THEN 1 ELSE 0 END) FROM anomaly_flags
        UNION ALL
        SELECT 'speed_over_100_mph', SUM(CASE WHEN speed_over_100_mph THEN 1 ELSE 0 END) FROM anomaly_flags
        UNION ALL
        SELECT 'negative_fare_amount', SUM(CASE WHEN negative_fare_amount THEN 1 ELSE 0 END) FROM anomaly_flags
        UNION ALL
        SELECT 'negative_total_amount', SUM(CASE WHEN negative_total_amount THEN 1 ELSE 0 END) FROM anomaly_flags
        ORDER BY row_count DESC
        """,
        out_dir / f"{prefix}_anomaly_counts.csv",
        [start_date, end_date, args.taxi_type, args.year, args.month],
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()