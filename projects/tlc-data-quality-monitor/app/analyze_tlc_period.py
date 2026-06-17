from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate multi-month period reports from persistent clean TLC tables.")
    p.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "db" / "tlc.duckdb")
    p.add_argument("--taxi-type", default="yellow", choices=["yellow"])
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--start-month", type=int, required=True)
    p.add_argument("--end-month", type=int, required=True)
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "reports")
    return p.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_csv(con: duckdb.DuckDBPyConnection, sql: str, path: Path, params: list | None = None) -> None:
    df = con.execute(sql, params or []).fetchdf()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved: {path} rows={len(df)}")


def fetch_one(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> dict:
    df = con.execute(sql, params or []).fetchdf()
    return {} if len(df) == 0 else df.to_dict(orient="records")[0]


def main() -> None:
    args = parse_args()
    if not (1 <= args.start_month <= 12 and 1 <= args.end_month <= 12):
        raise ValueError("months must be 1-12")
    if args.end_month < args.start_month:
        raise ValueError("--end-month must be >= --start-month")

    db_path = resolve_path(args.db)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    con = duckdb.connect(str(db_path))
    prefix = f"{args.taxi_type}_{args.year}_{args.start_month:02d}_{args.end_month:02d}"
    params = [args.taxi_type, args.year, args.start_month, args.end_month]

    coverage = con.execute("""
        SELECT data_year, data_month, COUNT(*) AS clean_trip_count, COUNT(DISTINCT pickup_date) AS active_pickup_days
        FROM clean_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month BETWEEN ? AND ?
        GROUP BY data_year, data_month
        ORDER BY data_year, data_month
    """, params).fetchdf()

    if len(coverage) == 0:
        raise RuntimeError("No clean trips found for requested period. Ingest those months first.")

    period_summary = fetch_one(con, """
        SELECT
            COUNT(*) AS clean_trip_count,
            COUNT(DISTINCT pickup_date) AS active_pickup_days,
            MIN(pickup_datetime) AS min_pickup_datetime,
            MAX(pickup_datetime) AS max_pickup_datetime,
            SUM(total_amount) AS gross_total_amount,
            SUM(fare_amount) AS gross_fare_amount,
            SUM(tip_amount) AS gross_tip_amount,
            AVG(total_amount) AS avg_total_amount,
            MEDIAN(total_amount) AS median_total_amount,
            AVG(trip_distance) AS avg_trip_distance_miles,
            MEDIAN(trip_distance) AS median_trip_distance_miles,
            AVG(duration_min) AS avg_duration_min,
            MEDIAN(duration_min) AS median_duration_min,
            AVG(speed_mph) AS avg_speed_mph,
            MEDIAN(speed_mph) AS median_speed_mph
        FROM clean_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month BETWEEN ? AND ?
    """, params)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "taxi_type": args.taxi_type,
        "year": args.year,
        "start_month": args.start_month,
        "end_month": args.end_month,
        "coverage": coverage.to_dict(orient="records"),
        "period_summary": period_summary,
        "normalization_note": "Period hourly reports use average pickups per active pickup day, not raw totals.",
    }
    summary_path = out_dir / f"{prefix}_period_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"saved: {summary_path}")

    save_csv(con, """
        SELECT
            data_year,
            data_month,
            COUNT(*) AS clean_trip_count,
            COUNT(DISTINCT pickup_date) AS active_pickup_days,
            SUM(total_amount) AS gross_total_amount,
            SUM(total_amount) / COUNT(DISTINCT pickup_date) AS avg_gross_total_per_day,
            COUNT(*)::DOUBLE / COUNT(DISTINCT pickup_date) AS avg_trips_per_day,
            AVG(total_amount) AS avg_total_amount,
            AVG(trip_distance) AS avg_trip_distance_miles,
            AVG(duration_min) AS avg_duration_min
        FROM clean_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month BETWEEN ? AND ?
        GROUP BY data_year, data_month
        ORDER BY data_year, data_month
    """, out_dir / f"{prefix}_monthly_trend.csv", params)

    save_csv(con, """
        WITH period_days AS (
            SELECT COUNT(DISTINCT pickup_date) AS n_days
            FROM clean_yellow_trips
            WHERE taxi_type = ? AND data_year = ? AND data_month BETWEEN ? AND ?
        )
        SELECT
            t.pickup_hour,
            COUNT(*) AS total_trip_count,
            COUNT(*)::DOUBLE / MAX(d.n_days) AS avg_pickups_per_day,
            SUM(t.total_amount) AS gross_total_amount,
            SUM(t.total_amount) / MAX(d.n_days) AS avg_gross_total_per_day,
            AVG(t.total_amount) AS avg_total_amount,
            AVG(t.trip_distance) AS avg_trip_distance_miles,
            AVG(t.duration_min) AS avg_duration_min
        FROM clean_yellow_trips t
        CROSS JOIN period_days d
        WHERE t.taxi_type = ? AND t.data_year = ? AND t.data_month BETWEEN ? AND ?
        GROUP BY t.pickup_hour
        ORDER BY t.pickup_hour
    """, out_dir / f"{prefix}_hourly_profile_avg_day.csv", params + params)

    save_csv(con, """
        WITH period_days AS (
            SELECT COUNT(DISTINCT pickup_date) AS n_days
            FROM clean_yellow_trips
            WHERE taxi_type = ? AND data_year = ? AND data_month BETWEEN ? AND ?
        )
        SELECT
            t.pickup_location_id,
            COALESCE(z.borough, 'Unknown') AS pickup_borough,
            COALESCE(z.zone, 'Unknown') AS pickup_zone,
            COUNT(*) AS total_trip_count,
            COUNT(*)::DOUBLE / MAX(d.n_days) AS avg_pickups_per_day,
            SUM(t.total_amount) AS gross_total_amount,
            SUM(t.total_amount) / MAX(d.n_days) AS avg_gross_total_per_day
        FROM clean_yellow_trips t
        CROSS JOIN period_days d
        LEFT JOIN dim_taxi_zones z ON t.pickup_location_id = z.location_id
        WHERE t.taxi_type = ? AND t.data_year = ? AND t.data_month BETWEEN ? AND ?
        GROUP BY t.pickup_location_id, pickup_borough, pickup_zone
        ORDER BY avg_pickups_per_day DESC
        LIMIT 100
    """, out_dir / f"{prefix}_top_pickup_zones_avg_day.csv", params + params)

    save_csv(con, """
        WITH period_days AS (
            SELECT COUNT(DISTINCT pickup_date) AS n_days
            FROM clean_yellow_trips
            WHERE taxi_type = ? AND data_year = ? AND data_month BETWEEN ? AND ?
        )
        SELECT
            t.pickup_location_id,
            COALESCE(z.borough, 'Unknown') AS pickup_borough,
            COALESCE(z.zone, 'Unknown') AS pickup_zone,
            t.pickup_hour,
            COUNT(*) AS total_trip_count,
            COUNT(*)::DOUBLE / MAX(d.n_days) AS avg_pickups_per_day_at_hour,
            SUM(t.total_amount) / MAX(d.n_days) AS avg_gross_total_per_day_at_hour
        FROM clean_yellow_trips t
        CROSS JOIN period_days d
        LEFT JOIN dim_taxi_zones z ON t.pickup_location_id = z.location_id
        WHERE t.taxi_type = ? AND t.data_year = ? AND t.data_month BETWEEN ? AND ?
        GROUP BY t.pickup_location_id, pickup_borough, pickup_zone, t.pickup_hour
        ORDER BY t.pickup_hour, avg_pickups_per_day_at_hour DESC
    """, out_dir / f"{prefix}_zone_hour_profile_avg_day.csv", params + params)

    print("\nPeriod analysis complete.")


if __name__ == "__main__":
    main()
