from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from build_pickup_density_hybrid_taxi_brand import (
    PROJECT_ROOT,
    attach_hourly_series_to_zones,
    load_zone_geojson,
    resolve_path,
    write_hybrid_html,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a taxi-branded period animation using average pickups per active day by hour."
        )
    )
    p.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "db" / "tlc.duckdb")
    p.add_argument("--taxi-type", default="yellow", choices=["yellow"])
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--start-month", type=int, required=True)
    p.add_argument("--end-month", type=int, required=True)
    p.add_argument("--zones", type=Path, default=PROJECT_ROOT / "data" / "raw" / "tlc" / "taxi_zones_4326.parquet")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max-elevation", type=float, default=1800.0)
    p.add_argument("--start-hour", type=int, default=6)
    p.add_argument("--ms-per-hour", type=int, default=650)
    return p.parse_args()


def load_period_counts(db_path: Path, taxi_type: str, year: int, start_month: int, end_month: int):
    if not db_path.exists():
        raise FileNotFoundError(f"Missing DuckDB file: {db_path}")
    con = duckdb.connect(str(db_path))

    n_days = con.execute(
        """
        SELECT COUNT(DISTINCT pickup_date)
        FROM clean_yellow_trips
        WHERE taxi_type = ? AND data_year = ? AND data_month BETWEEN ? AND ?
        """,
        [taxi_type, year, start_month, end_month],
    ).fetchone()[0]

    if not n_days:
        raise RuntimeError("No clean trips found for selected period. Ingest those months first.")

    df = con.execute(
        """
        WITH period_days AS (
            SELECT COUNT(DISTINCT pickup_date) AS n_days
            FROM clean_yellow_trips
            WHERE taxi_type = ? AND data_year = ? AND data_month BETWEEN ? AND ?
        )
        SELECT
            pickup_location_id AS location_id,
            pickup_hour,
            COUNT(*)::DOUBLE / MAX(d.n_days) AS trip_count,
            SUM(total_amount)::DOUBLE / MAX(d.n_days) AS gross_total_amount,
            AVG(total_amount) AS avg_total_amount
        FROM clean_yellow_trips t
        CROSS JOIN period_days d
        WHERE taxi_type = ? AND data_year = ? AND data_month BETWEEN ? AND ?
        GROUP BY pickup_location_id, pickup_hour
        ORDER BY pickup_hour, trip_count DESC
        """,
        [taxi_type, year, start_month, end_month, taxi_type, year, start_month, end_month],
    ).fetchdf()

    return df, int(n_days)


def patch_html_for_period(output_path: Path, year: int, start_month: int, end_month: int, active_days: int) -> None:
    html = output_path.read_text(encoding="utf-8")
    old_subtitle = f"{year}-{start_month:02d} · pickup density by TLC taxi zone"
    new_subtitle = (
        f"{year}-{start_month:02d} TO {year}-{end_month:02d} · "
        f"AVG PICKUPS PER DAY BY HOUR · {active_days} ACTIVE DAYS"
    )
    html = html.replace(old_subtitle, new_subtitle)
    html = html.replace("Pickups:", "Avg pickups/day:")
    html = html.replace("Gross:", "Avg gross/day:")
    html = html.replace(" pickups`;", " avg pickups/day`;')") if False else html
    html = html.replace(" pickups`,", " avg pickups/day`,")
    html = html.replace(" pickup density", " avg pickup density")
    html = html.replace("NYC TLC trip records · taxi zone polygons", "NYC TLC trip records · avg daily hourly pickups")
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.start_month < 1 or args.start_month > 12:
        raise ValueError("--start-month must be 1-12")
    if args.end_month < 1 or args.end_month > 12:
        raise ValueError("--end-month must be 1-12")
    if args.end_month < args.start_month:
        raise ValueError("--end-month must be >= --start-month")

    db_path = resolve_path(args.db)
    zones_path = resolve_path(args.zones)

    if args.output is None:
        output_path = (
            PROJECT_ROOT
            / "outputs"
            / "reports"
            / "maps"
            / f"{args.taxi_type}_{args.year}_{args.start_month:02d}_{args.end_month:02d}_pickup_density_period_hybrid.html"
        )
    else:
        output_path = resolve_path(args.output)

    zones_geojson = load_zone_geojson(zones_path)
    counts_df, active_days = load_period_counts(
        db_path=db_path,
        taxi_type=args.taxi_type,
        year=args.year,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    enriched_zones_geojson, max_trip_count = attach_hourly_series_to_zones(zones_geojson, counts_df)

    # Reuse the final monthly branded renderer. Then patch labels/subtitle for period meaning.
    write_hybrid_html(
        output_path=output_path,
        zones_geojson=enriched_zones_geojson,
        taxi_type=args.taxi_type,
        year=args.year,
        month=args.start_month,
        max_trip_count=max_trip_count,
        max_elevation=args.max_elevation,
        start_hour=args.start_hour,
        ms_per_hour=args.ms_per_hour,
    )
    patch_html_for_period(output_path, args.year, args.start_month, args.end_month, active_days)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
