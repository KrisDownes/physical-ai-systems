from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Citi Bike station reliability intervals from status snapshots."
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "urban_pulse.duckdb",
    )

    parser.add_argument(
        "--max-gap-min",
        type=float,
        default=3.0,
        help=(
            "Maximum allowed gap between source updates. Larger gaps are treated "
            "as missing collection time and excluded from interval calculations."
        ),
    )

    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=None,
        help=(
            "If provided, station reliability reports use only intervals from the "
            "latest N hours relative to the newest interval_end_utc."
        ),
    )

    parser.add_argument(
        "--min-observed-min",
        type=float,
        default=30.0,
        help="Minimum observed minutes required for a station to appear in the reliability report.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    db_path = resolve_path(args.db)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")

    max_gap_sec = args.max_gap_min * 60.0

    con = duckdb.connect(str(db_path))

    print("Building Citi Bike reliability intervals")
    print(f"db:               {db_path}")
    print(f"max_gap_min:      {args.max_gap_min}")
    print(f"lookback_hours:   {args.lookback_hours}")
    print(f"min_observed_min: {args.min_observed_min}")

    con.execute("DROP TABLE IF EXISTS fact_citibike_station_status_intervals;")

    con.execute(
        """
        CREATE TABLE fact_citibike_station_status_intervals AS
        WITH ordered AS (
            SELECT
                station_id,
                source_last_updated_utc AS interval_start_utc,
                LEAD(source_last_updated_utc) OVER (
                    PARTITION BY station_id
                    ORDER BY source_last_updated_utc
                ) AS interval_end_utc,

                status_class,
                num_bikes_available,
                num_ebikes_available,
                num_bikes_disabled,
                num_docks_available,
                num_docks_disabled,
                is_installed,
                is_renting,
                is_returning
            FROM fact_citibike_station_status
        ),

        intervals AS (
            SELECT
                *,
                DATE_DIFF('second', interval_start_utc, interval_end_utc) AS duration_sec
            FROM ordered
            WHERE interval_end_utc IS NOT NULL
        )

        SELECT
            station_id,
            interval_start_utc,
            interval_end_utc,
            duration_sec / 60.0 AS duration_min,
            status_class,
            num_bikes_available,
            num_ebikes_available,
            num_bikes_disabled,
            num_docks_available,
            num_docks_disabled,
            is_installed,
            is_renting,
            is_returning
        FROM intervals
        WHERE
            duration_sec > 0
            AND duration_sec <= ?;
        """,
        [max_gap_sec],
    )

    interval_count = con.execute(
        """
        SELECT COUNT(*)
        FROM fact_citibike_station_status_intervals
        """
    ).fetchone()[0]

    print(f"\ninterval rows: {interval_count}")

    interval_summary = con.execute(
        """
        SELECT
            status_class,
            COUNT(*) AS interval_count,
            ROUND(SUM(duration_min), 2) AS total_station_minutes,
            ROUND(AVG(duration_min), 2) AS avg_interval_min
        FROM fact_citibike_station_status_intervals
        GROUP BY status_class
        ORDER BY total_station_minutes DESC
        """
    ).fetchdf()

    print("\nInterval summary by status:")
    print(interval_summary)

    interval_summary_path = out_dir / "citibike_interval_summary_by_status.csv"
    interval_summary.to_csv(interval_summary_path, index=False)
    print(f"\nsaved: {interval_summary_path}")

    latest_interval_end = con.execute(
        """
        SELECT MAX(interval_end_utc)
        FROM fact_citibike_station_status_intervals
        """
    ).fetchone()[0]

    lookback_start = None
    if args.lookback_hours is not None:
        lookback_start = latest_interval_end - timedelta(hours=args.lookback_hours)
        print(f"\nUsing lookback window:")
        print(f"latest_interval_end: {latest_interval_end}")
        print(f"lookback_start:      {lookback_start}")

    lookback_filter_sql = ""
    params: list[object] = []

    if lookback_start is not None:
        lookback_filter_sql = "AND i.interval_start_utc >= ?"
        params.append(lookback_start)

    params.append(args.min_observed_min)

    station_reliability = con.execute(
        f"""
        WITH filtered_intervals AS (
            SELECT i.*
            FROM fact_citibike_station_status_intervals i
            WHERE 1 = 1
            {lookback_filter_sql}
        )

        SELECT
            i.station_id,
            COALESCE(s.name, 'Unknown') AS station_name,
            COALESCE(s.capacity, 0) AS capacity,
            s.lat,
            s.lon,

            ROUND(SUM(i.duration_min), 2) AS observed_minutes,

            ROUND(
                SUM(CASE WHEN i.status_class = 'available' THEN i.duration_min ELSE 0 END),
                2
            ) AS available_minutes,

            ROUND(
                SUM(CASE WHEN i.status_class = 'empty' THEN i.duration_min ELSE 0 END),
                2
            ) AS empty_minutes,

            ROUND(
                SUM(CASE WHEN i.status_class = 'full' THEN i.duration_min ELSE 0 END),
                2
            ) AS full_minutes,

            ROUND(
                SUM(CASE WHEN i.status_class = 'offline' THEN i.duration_min ELSE 0 END),
                2
            ) AS offline_minutes,

            ROUND(
                SUM(CASE WHEN i.status_class = 'available' THEN i.duration_min ELSE 0 END)
                / NULLIF(SUM(i.duration_min), 0),
                4
            ) AS availability_ratio,

            ROUND(
                SUM(CASE WHEN i.status_class IN ('empty', 'offline') THEN i.duration_min ELSE 0 END)
                / NULLIF(SUM(i.duration_min), 0),
                4
            ) AS rental_failure_ratio,

            ROUND(
                SUM(CASE WHEN i.status_class IN ('full', 'offline') THEN i.duration_min ELSE 0 END)
                / NULLIF(SUM(i.duration_min), 0),
                4
            ) AS return_failure_ratio,

            ROUND(
                SUM(CASE WHEN i.status_class IN ('empty', 'full', 'offline') THEN i.duration_min ELSE 0 END)
                / NULLIF(SUM(i.duration_min), 0),
                4
            ) AS any_failure_ratio
        FROM filtered_intervals i
        LEFT JOIN dim_citibike_stations s
            ON i.station_id = s.station_id
        WHERE
            i.status_class != 'not_installed'
            AND COALESCE(s.capacity, 0) > 0
        GROUP BY
            i.station_id,
            station_name,
            capacity,
            s.lat,
            s.lon
        HAVING SUM(i.duration_min) >= ?
        ORDER BY any_failure_ratio DESC, observed_minutes DESC
        """,
        params,
    ).fetchdf()

    station_path = out_dir / "citibike_station_reliability_summary.csv"
    station_reliability.to_csv(station_path, index=False)
    print(f"saved: {station_path}")

    print("\nWorst stations by any_failure_ratio:")
    print(station_reliability.head(20))


if __name__ == "__main__":
    main()
