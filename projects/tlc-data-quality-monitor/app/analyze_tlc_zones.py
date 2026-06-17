from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate zone-enriched NYC TLC reports using dim_taxi_zones."
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "tlc.duckdb",
        help="DuckDB database path.",
    )

    parser.add_argument("--taxi-type", default="yellow", choices=["yellow"])
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)

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


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return (
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
        > 0
    )


def save_csv(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    path: Path,
    params: list | None = None,
) -> None:
    df = con.execute(sql, params or []).fetchdf()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved: {path} rows={len(df)}")


def main() -> None:
    args = parse_args()

    db_path = resolve_path(args.db)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    con = duckdb.connect(str(db_path))

    if not table_exists(con, "clean_yellow_trips"):
        raise RuntimeError("Missing table: clean_yellow_trips. Run ingest_tlc_month.py first.")

    if not table_exists(con, "dim_taxi_zones"):
        raise RuntimeError("Missing table: dim_taxi_zones. Run load_taxi_zones.py first.")

    prefix = f"{args.taxi_type}_{args.year}_{args.month:02d}"

    print("TLC zone analysis")
    print(f"db:        {db_path}")
    print(f"taxi_type: {args.taxi_type}")
    print(f"year:      {args.year}")
    print(f"month:     {args.month:02d}")

    params = [args.taxi_type, args.year, args.month]

    save_csv(
        con,
        """
        SELECT
            t.pickup_location_id,
            COALESCE(z.borough, 'Unknown') AS pickup_borough,
            COALESCE(z.zone, 'Unknown') AS pickup_zone,
            COALESCE(z.service_zone, 'Unknown') AS pickup_service_zone,
            COUNT(*) AS trip_count,
            SUM(t.total_amount) AS gross_total_amount,
            AVG(t.total_amount) AS avg_total_amount,
            AVG(t.fare_amount) AS avg_fare_amount,
            AVG(t.tip_amount) AS avg_tip_amount,
            AVG(t.trip_distance) AS avg_trip_distance_miles,
            AVG(t.duration_min) AS avg_duration_min,
            AVG(t.speed_mph) AS avg_speed_mph
        FROM clean_yellow_trips t
        LEFT JOIN dim_taxi_zones z
            ON t.pickup_location_id = z.location_id
        WHERE t.taxi_type = ? AND t.data_year = ? AND t.data_month = ?
        GROUP BY
            t.pickup_location_id,
            pickup_borough,
            pickup_zone,
            pickup_service_zone
        ORDER BY trip_count DESC
        LIMIT 50
        """,
        out_dir / f"{prefix}_top_pickup_zones_named.csv",
        params,
    )

    save_csv(
        con,
        """
        SELECT
            t.dropoff_location_id,
            COALESCE(z.borough, 'Unknown') AS dropoff_borough,
            COALESCE(z.zone, 'Unknown') AS dropoff_zone,
            COALESCE(z.service_zone, 'Unknown') AS dropoff_service_zone,
            COUNT(*) AS trip_count,
            SUM(t.total_amount) AS gross_total_amount,
            AVG(t.total_amount) AS avg_total_amount,
            AVG(t.trip_distance) AS avg_trip_distance_miles,
            AVG(t.duration_min) AS avg_duration_min
        FROM clean_yellow_trips t
        LEFT JOIN dim_taxi_zones z
            ON t.dropoff_location_id = z.location_id
        WHERE t.taxi_type = ? AND t.data_year = ? AND t.data_month = ?
        GROUP BY
            t.dropoff_location_id,
            dropoff_borough,
            dropoff_zone,
            dropoff_service_zone
        ORDER BY trip_count DESC
        LIMIT 50
        """,
        out_dir / f"{prefix}_top_dropoff_zones_named.csv",
        params,
    )

    save_csv(
        con,
        """
        SELECT
            t.pickup_location_id,
            COALESCE(pu.borough, 'Unknown') AS pickup_borough,
            COALESCE(pu.zone, 'Unknown') AS pickup_zone,
            t.dropoff_location_id,
            COALESCE(doz.borough, 'Unknown') AS dropoff_borough,
            COALESCE(doz.zone, 'Unknown') AS dropoff_zone,
            COUNT(*) AS trip_count,
            SUM(t.total_amount) AS gross_total_amount,
            AVG(t.total_amount) AS avg_total_amount,
            AVG(t.trip_distance) AS avg_trip_distance_miles,
            AVG(t.duration_min) AS avg_duration_min
        FROM clean_yellow_trips t
        LEFT JOIN dim_taxi_zones pu
            ON t.pickup_location_id = pu.location_id
        LEFT JOIN dim_taxi_zones doz
            ON t.dropoff_location_id = doz.location_id
        WHERE t.taxi_type = ? AND t.data_year = ? AND t.data_month = ?
        GROUP BY
            t.pickup_location_id,
            pickup_borough,
            pickup_zone,
            t.dropoff_location_id,
            dropoff_borough,
            dropoff_zone
        ORDER BY trip_count DESC
        LIMIT 100
        """,
        out_dir / f"{prefix}_top_routes_named.csv",
        params,
    )

    save_csv(
        con,
        """
        SELECT
            COALESCE(z.borough, 'Unknown') AS pickup_borough,
            t.pickup_hour,
            COUNT(*) AS trip_count,
            SUM(t.total_amount) AS gross_total_amount,
            AVG(t.total_amount) AS avg_total_amount,
            AVG(t.trip_distance) AS avg_trip_distance_miles,
            AVG(t.duration_min) AS avg_duration_min
        FROM clean_yellow_trips t
        LEFT JOIN dim_taxi_zones z
            ON t.pickup_location_id = z.location_id
        WHERE t.taxi_type = ? AND t.data_year = ? AND t.data_month = ?
        GROUP BY pickup_borough, t.pickup_hour
        ORDER BY pickup_borough, t.pickup_hour
        """,
        out_dir / f"{prefix}_borough_hour_demand.csv",
        params,
    )

    print("\nZone analysis complete.")


if __name__ == "__main__":
    main()