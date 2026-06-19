from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze NYC DOT traffic speed observations."
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "urban_pulse.duckdb",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports",
    )

    parser.add_argument(
        "--freshness-min",
        type=float,
        default=15.0,
        help="Segments older than this many minutes behind the latest DOT timestamp are flagged stale.",
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

    con = duckdb.connect(str(db_path))

    print("Analyzing DOT traffic speeds")
    print(f"db:            {db_path}")
    print(f"freshness_min: {args.freshness_min}")

    current_segments = con.execute(
        """
        WITH cleaned AS (
            SELECT
                f.data_as_of,
                f.link_id,
                f.speed_mph,
                f.travel_time_sec,
                f.status,
                COALESCE(f.borough, s.borough) AS borough,
                COALESCE(f.link_name, s.link_name) AS link_name,
                COALESCE(f.owner, s.owner) AS owner,
                s.start_lat,
                s.start_lon,
                s.end_lat,
                s.end_lon,
                s.point_count
            FROM fact_dot_traffic_speeds f
            LEFT JOIN dim_dot_traffic_segments s
                ON f.link_id = s.link_id
            WHERE
                f.speed_mph IS NOT NULL
                AND f.speed_mph > 0
        ),

        latest_clock AS (
            SELECT MAX(data_as_of) AS latest_data_as_of
            FROM cleaned
        ),

        baseline AS (
            SELECT
                link_id,
                COUNT(*) AS observations,
                ROUND(AVG(speed_mph), 2) AS avg_speed_mph,
                ROUND(MEDIAN(speed_mph), 2) AS median_speed_mph,
                ROUND(QUANTILE_CONT(speed_mph, 0.75), 2) AS p75_speed_mph,
                ROUND(QUANTILE_CONT(speed_mph, 0.90), 2) AS p90_speed_mph,
                ROUND(MIN(speed_mph), 2) AS min_speed_mph,
                ROUND(MAX(speed_mph), 2) AS max_speed_mph
            FROM cleaned
            GROUP BY link_id
        ),

        ranked_current AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY link_id
                    ORDER BY data_as_of DESC
                ) AS rn
            FROM cleaned
        ),

        current AS (
            SELECT *
            FROM ranked_current
            WHERE rn = 1
        ),

        scored AS (
            SELECT
                c.data_as_of,
                lc.latest_data_as_of,
                DATE_DIFF('minute', c.data_as_of, lc.latest_data_as_of)
                    AS minutes_behind_latest,

                c.link_id,
                c.borough,
                c.link_name,
                c.owner,
                c.status,
                c.speed_mph AS current_speed_mph,
                c.travel_time_sec,
                c.start_lat,
                c.start_lon,
                c.end_lat,
                c.end_lon,
                c.point_count,

                b.observations,
                b.avg_speed_mph,
                b.median_speed_mph,
                b.p75_speed_mph,
                b.p90_speed_mph,
                b.min_speed_mph,
                b.max_speed_mph,

                CASE
                    WHEN b.p75_speed_mph > 0
                    THEN GREATEST(
                        0.0,
                        LEAST(1.0, 1.0 - (c.speed_mph / b.p75_speed_mph))
                    )
                    ELSE NULL
                END AS slowdown_ratio
            FROM current c
            LEFT JOIN baseline b
                ON c.link_id = b.link_id
            CROSS JOIN latest_clock lc
        )

        SELECT
            data_as_of,
            latest_data_as_of,
            minutes_behind_latest,
            CASE
                WHEN minutes_behind_latest <= ? THEN TRUE
                ELSE FALSE
            END AS is_fresh,

            link_id,
            borough,
            link_name,
            owner,
            status,

            current_speed_mph,
            travel_time_sec,

            observations,
            avg_speed_mph,
            median_speed_mph,
            p75_speed_mph,
            p90_speed_mph,
            min_speed_mph,
            max_speed_mph,

            ROUND(slowdown_ratio, 4) AS slowdown_ratio,
            ROUND(100.0 * slowdown_ratio, 1) AS traffic_stress_score,

            start_lat,
            start_lon,
            end_lat,
            end_lon,
            point_count
        FROM scored
        ORDER BY traffic_stress_score DESC NULLS LAST, current_speed_mph ASC
        """,
        [args.freshness_min],
    ).fetchdf()

    current_path = out_dir / "dot_traffic_current_segments.csv"
    current_segments.to_csv(current_path, index=False)

    fresh = current_segments[current_segments["is_fresh"] == True].copy()

    borough_summary = (
        fresh.groupby("borough", dropna=False)
        .agg(
            segment_count=("link_id", "count"),
            avg_current_speed_mph=("current_speed_mph", "mean"),
            avg_baseline_p75_mph=("p75_speed_mph", "mean"),
            avg_slowdown_ratio=("slowdown_ratio", "mean"),
            avg_traffic_stress_score=("traffic_stress_score", "mean"),
            stale_count=("is_fresh", lambda x: int((~x).sum())),
        )
        .reset_index()
    )

    numeric_cols = [
        "avg_current_speed_mph",
        "avg_baseline_p75_mph",
        "avg_slowdown_ratio",
        "avg_traffic_stress_score",
    ]
    borough_summary[numeric_cols] = borough_summary[numeric_cols].round(3)

    borough_summary = borough_summary.sort_values(
        "avg_traffic_stress_score",
        ascending=False,
    )

    borough_path = out_dir / "dot_traffic_borough_summary.csv"
    borough_summary.to_csv(borough_path, index=False)

    stressed_segments = fresh.sort_values(
        ["traffic_stress_score", "current_speed_mph"],
        ascending=[False, True],
    ).head(30)

    stressed_path = out_dir / "dot_traffic_stressed_segments.csv"
    stressed_segments.to_csv(stressed_path, index=False)

    print("\nSaved:")
    print(f"current segments: {current_path}")
    print(f"borough summary:  {borough_path}")
    print(f"stressed links:   {stressed_path}")

    print("\nCurrent segment coverage:")
    print(
        current_segments[
            [
                "link_id",
                "data_as_of",
                "minutes_behind_latest",
                "is_fresh",
                "current_speed_mph",
                "p75_speed_mph",
                "traffic_stress_score",
            ]
        ].describe(include="all")
    )

    print("\nBorough summary:")
    print(borough_summary)

    print("\nMost stressed fresh segments:")
    print(
        stressed_segments[
            [
                "borough",
                "link_name",
                "data_as_of",
                "current_speed_mph",
                "p75_speed_mph",
                "traffic_stress_score",
            ]
        ].head(20)
    )


if __name__ == "__main__":
    main()
