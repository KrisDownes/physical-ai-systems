from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EARTH_RADIUS_M = 6_371_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse Citi Bike station reliability with nearby DOT traffic stress."
    )

    parser.add_argument(
        "--bike-csv",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "reports"
        / "citibike_station_reliability_summary.csv",
    )

    parser.add_argument(
        "--traffic-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "dot_traffic_current_segments.csv",
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
        "--radius-m",
        type=float,
        default=750.0,
        help="Road segments within this distance of a station are used for local traffic stress.",
    )

    parser.add_argument(
        "--min-observed-min",
        type=float,
        default=30.0,
        help="Minimum Citi Bike observation minutes required.",
    )

    parser.add_argument(
        "--bike-weight",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--traffic-weight",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--include-stale-traffic",
        action="store_true",
        help="Use stale DOT segments too. Default uses only fresh traffic rows.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_link_points(value: Any) -> list[tuple[float, float]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    points: list[tuple[float, float]] = []

    for token in str(value).split():
        try:
            lat_str, lon_str = token.split(",", 1)
            points.append((float(lat_str), float(lon_str)))
        except Exception:
            continue

    return points


def latlon_to_xy_m(
    lat: float,
    lon: float,
    *,
    origin_lat: float,
    origin_lon: float,
) -> tuple[float, float]:
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    origin_lat_rad = math.radians(origin_lat)
    origin_lon_rad = math.radians(origin_lon)

    x = EARTH_RADIUS_M * (lon_rad - origin_lon_rad) * math.cos(origin_lat_rad)
    y = EARTH_RADIUS_M * (lat_rad - origin_lat_rad)

    return x, y


def point_to_segment_distance_m(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx = bx - ax
    aby = by - ay

    apx = px - ax
    apy = py - ay

    ab_len_sq = abx * abx + aby * aby

    if ab_len_sq == 0:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab_len_sq
    t = max(0.0, min(1.0, t))

    closest_x = ax + t * abx
    closest_y = ay + t * aby

    return math.hypot(px - closest_x, py - closest_y)


def station_to_polyline_distance_m(
    station_lat: float,
    station_lon: float,
    points: list[tuple[float, float]],
) -> float | None:
    if not points:
        return None

    if len(points) == 1:
        x, y = latlon_to_xy_m(
            points[0][0],
            points[0][1],
            origin_lat=station_lat,
            origin_lon=station_lon,
        )
        return math.hypot(x, y)

    projected = [
        latlon_to_xy_m(
            lat,
            lon,
            origin_lat=station_lat,
            origin_lon=station_lon,
        )
        for lat, lon in points
    ]

    distances = []

    for idx in range(len(projected) - 1):
        ax, ay = projected[idx]
        bx, by = projected[idx + 1]

        distances.append(
            point_to_segment_distance_m(
                0.0,
                0.0,
                ax,
                ay,
                bx,
                by,
            )
        )

    return min(distances)


def is_true_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def load_traffic_with_geometry(
    traffic_csv: Path,
    db_path: Path,
    *,
    include_stale: bool,
) -> pd.DataFrame:
    traffic = pd.read_csv(traffic_csv)

    traffic["link_id"] = traffic["link_id"].astype(str)
    traffic["traffic_stress_score"] = pd.to_numeric(
        traffic["traffic_stress_score"],
        errors="coerce",
    )

    traffic = traffic.dropna(subset=["traffic_stress_score"]).copy()

    if not include_stale and "is_fresh" in traffic.columns:
        traffic = traffic[is_true_series(traffic["is_fresh"])].copy()

    if db_path.exists():
        con = duckdb.connect(str(db_path))

        geom = con.execute(
            """
            SELECT
                link_id,
                link_points
            FROM dim_dot_traffic_segments
            """
        ).fetchdf()

        geom["link_id"] = geom["link_id"].astype(str)

        traffic = traffic.merge(
            geom,
            on="link_id",
            how="left",
        )
    else:
        traffic["link_points"] = None

    fallback_points = []

    for _, row in traffic.iterrows():
        points = parse_link_points(row.get("link_points"))

        if not points:
            if pd.notna(row.get("start_lat")) and pd.notna(row.get("start_lon")):
                points.append((float(row["start_lat"]), float(row["start_lon"])))

            if pd.notna(row.get("end_lat")) and pd.notna(row.get("end_lon")):
                points.append((float(row["end_lat"]), float(row["end_lon"])))

        fallback_points.append(points)

    traffic["polyline_points"] = fallback_points

    traffic = traffic[traffic["polyline_points"].map(len) > 0].copy()

    return traffic


def classify_station(
    bike_score: float,
    traffic_score: float | None,
) -> str:
    if traffic_score is None or pd.isna(traffic_score):
        if bike_score >= 50:
            return "bike stress only / no nearby detector"
        return "no nearby traffic detector"

    if bike_score >= 50 and traffic_score >= 50:
        return "high bike + high traffic stress"

    if bike_score >= 50 and traffic_score < 25:
        return "bike stress only"

    if bike_score < 25 and traffic_score >= 50:
        return "traffic stress only"

    if bike_score < 25 and traffic_score < 25:
        return "low local stress"

    return "mixed / moderate stress"


def main() -> None:
    args = parse_args()

    bike_csv = resolve_path(args.bike_csv)
    traffic_csv = resolve_path(args.traffic_csv)
    db_path = resolve_path(args.db)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bike_csv.exists():
        raise FileNotFoundError(f"Citi Bike reliability CSV not found: {bike_csv}")

    if not traffic_csv.exists():
        raise FileNotFoundError(f"DOT traffic current segment CSV not found: {traffic_csv}")

    if args.bike_weight < 0 or args.traffic_weight < 0:
        raise ValueError("Weights must be non-negative.")

    weight_total = args.bike_weight + args.traffic_weight
    if weight_total <= 0:
        raise ValueError("At least one weight must be positive.")

    bike_weight = args.bike_weight / weight_total
    traffic_weight = args.traffic_weight / weight_total

    bike = pd.read_csv(bike_csv)
    bike = bike.dropna(subset=["lat", "lon"]).copy()
    bike = bike[bike["observed_minutes"] >= args.min_observed_min].copy()

    traffic = load_traffic_with_geometry(
        traffic_csv,
        db_path,
        include_stale=args.include_stale_traffic,
    )

    fused_rows = []

    for _, station in bike.iterrows():
        station_lat = float(station["lat"])
        station_lon = float(station["lon"])

        segment_matches = []

        for _, segment in traffic.iterrows():
            distance_m = station_to_polyline_distance_m(
                station_lat,
                station_lon,
                segment["polyline_points"],
            )

            if distance_m is None:
                continue

            segment_matches.append(
                {
                    "distance_m": distance_m,
                    "link_id": segment["link_id"],
                    "link_name": segment.get("link_name"),
                    "borough": segment.get("borough"),
                    "traffic_stress_score": float(segment["traffic_stress_score"]),
                    "current_speed_mph": segment.get("current_speed_mph"),
                    "p75_speed_mph": segment.get("p75_speed_mph"),
                }
            )

        segment_matches = sorted(segment_matches, key=lambda x: x["distance_m"])

        nearby = [
            match for match in segment_matches if match["distance_m"] <= args.radius_m
        ]

        nearest = segment_matches[0] if segment_matches else None

        local_traffic_stress_score = None
        nearest_traffic_stress_score = None
        nearest_traffic_distance_m = None
        nearest_traffic_link_name = None

        if nearest is not None:
            nearest_traffic_stress_score = nearest["traffic_stress_score"]
            nearest_traffic_distance_m = nearest["distance_m"]
            nearest_traffic_link_name = nearest["link_name"]

        if nearby:
            weights = np.array(
                [
                    math.exp(-match["distance_m"] / max(args.radius_m, 1.0))
                    for match in nearby
                ],
                dtype=float,
            )

            values = np.array(
                [match["traffic_stress_score"] for match in nearby],
                dtype=float,
            )

            local_traffic_stress_score = float(np.average(values, weights=weights))

        bike_unusable_score = 100.0 * float(station["any_failure_ratio"])
        unable_to_rent_score = 100.0 * float(station["rental_failure_ratio"])
        unable_to_dock_score = 100.0 * float(station["return_failure_ratio"])

        if local_traffic_stress_score is not None:
            fusion_mobility_stress_score = (
                bike_weight * bike_unusable_score
                + traffic_weight * local_traffic_stress_score
            )
        else:
            fusion_mobility_stress_score = None

        fused_rows.append(
            {
                "station_id": station["station_id"],
                "station_name": station["station_name"],
                "capacity": station["capacity"],
                "station_lat": station_lat,
                "station_lon": station_lon,
                "observed_minutes": station["observed_minutes"],

                "available_minutes": station["available_minutes"],
                "empty_minutes": station["empty_minutes"],
                "full_minutes": station["full_minutes"],
                "offline_minutes": station["offline_minutes"],

                "bike_unusable_score": round(bike_unusable_score, 2),
                "unable_to_rent_score": round(unable_to_rent_score, 2),
                "unable_to_dock_score": round(unable_to_dock_score, 2),

                "nearby_traffic_segment_count": len(nearby),
                "local_traffic_stress_score": (
                    round(local_traffic_stress_score, 2)
                    if local_traffic_stress_score is not None
                    else None
                ),
                "nearest_traffic_stress_score": (
                    round(nearest_traffic_stress_score, 2)
                    if nearest_traffic_stress_score is not None
                    else None
                ),
                "nearest_traffic_distance_m": (
                    round(nearest_traffic_distance_m, 1)
                    if nearest_traffic_distance_m is not None
                    else None
                ),
                "nearest_traffic_link_name": nearest_traffic_link_name,

                "fusion_mobility_stress_score": (
                    round(fusion_mobility_stress_score, 2)
                    if fusion_mobility_stress_score is not None
                    else None
                ),
                "stress_class": classify_station(
                    bike_unusable_score,
                    local_traffic_stress_score,
                ),
            }
        )

    fused = pd.DataFrame(fused_rows)

    fused_path = out_dir / "mobility_stress_station_fusion.csv"
    fused.to_csv(fused_path, index=False)

    covered = fused.dropna(subset=["local_traffic_stress_score"]).copy()

    top_fused = covered.sort_values(
        "fusion_mobility_stress_score",
        ascending=False,
    ).head(50)

    top_fused_path = out_dir / "mobility_stress_top_stations.csv"
    top_fused.to_csv(top_fused_path, index=False)

    class_summary = (
        fused.groupby("stress_class", dropna=False)
        .agg(
            station_count=("station_id", "count"),
            avg_bike_unusable_score=("bike_unusable_score", "mean"),
            avg_local_traffic_stress_score=("local_traffic_stress_score", "mean"),
            avg_fusion_mobility_stress_score=("fusion_mobility_stress_score", "mean"),
        )
        .reset_index()
    )

    summary_numeric_cols = [
        "avg_bike_unusable_score",
        "avg_local_traffic_stress_score",
        "avg_fusion_mobility_stress_score",
    ]

    class_summary[summary_numeric_cols] = class_summary[summary_numeric_cols].round(2)

    class_summary_path = out_dir / "mobility_stress_class_summary.csv"
    class_summary.to_csv(class_summary_path, index=False)

    correlation_rows = []

    if len(covered) >= 3:
        pearson = covered["bike_unusable_score"].corr(
            covered["local_traffic_stress_score"],
            method="pearson",
        )
        spearman = covered["bike_unusable_score"].corr(
            covered["local_traffic_stress_score"],
            method="spearman",
        )

        correlation_rows.append(
            {
                "radius_m": args.radius_m,
                "station_count": len(fused),
                "stations_with_nearby_traffic": len(covered),
                "coverage_ratio": len(covered) / len(fused),
                "pearson_bike_vs_traffic": pearson,
                "spearman_bike_vs_traffic": spearman,
            }
        )

    correlation = pd.DataFrame(correlation_rows)
    correlation_path = out_dir / "mobility_stress_correlation_summary.csv"
    correlation.to_csv(correlation_path, index=False)

    print("Mobility stress fusion complete.")
    print(f"bike_csv:     {bike_csv}")
    print(f"traffic_csv:  {traffic_csv}")
    print(f"radius_m:     {args.radius_m}")
    print(f"bike_weight:  {bike_weight:.2f}")
    print(f"traffic_weight: {traffic_weight:.2f}")

    print("\nSaved:")
    print(f"station fusion:      {fused_path}")
    print(f"top fused stations:  {top_fused_path}")
    print(f"class summary:       {class_summary_path}")
    print(f"correlation summary: {correlation_path}")

    print("\nCoverage:")
    print(f"stations:                    {len(fused)}")
    print(f"stations with nearby traffic: {len(covered)}")
    print(f"coverage ratio:              {len(covered) / len(fused):.2%}")

    print("\nCorrelation:")
    if not correlation.empty:
        print(correlation)
    else:
        print("Not enough stations with nearby traffic to compute correlation.")

    print("\nClass summary:")
    print(class_summary.sort_values("station_count", ascending=False))

    print("\nTop fused stress stations:")
    if not top_fused.empty:
        print(
            top_fused[
                [
                    "station_name",
                    "bike_unusable_score",
                    "local_traffic_stress_score",
                    "fusion_mobility_stress_score",
                    "nearby_traffic_segment_count",
                    "nearest_traffic_distance_m",
                    "stress_class",
                ]
            ].head(20)
        )
    else:
        print("No stations had nearby DOT traffic segments inside the radius.")


if __name__ == "__main__":
    main()
