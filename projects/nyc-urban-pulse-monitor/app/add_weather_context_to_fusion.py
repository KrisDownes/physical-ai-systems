from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EARTH_RADIUS_M = 6_371_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach weather context to station-level mobility stress fusion."
    )

    parser.add_argument(
        "--fusion-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "mobility_stress_station_fusion.csv",
    )

    parser.add_argument(
        "--weather-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "weather_current_context.csv",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "mobility_stress_station_fusion_weather.csv",
    )

    parser.add_argument(
        "--summary-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "mobility_stress_weather_summary.csv",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    )

    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def build_weather_adjusted_score(row: pd.Series) -> float | None:
    """
    mobility score = local bike/traffic stress
    weather score  = city/environmental stress

    Weather is a context amplifier, not a replacement for the physical mobility signals.
    """
    base = row.get("fusion_mobility_stress_score")

    if pd.isna(base):
        base = row.get("bike_unusable_score")

    weather = row.get("weather_stress_score")

    if pd.isna(base):
        return None

    if pd.isna(weather):
        return round(float(base), 2)

    score = 0.85 * float(base) + 0.15 * float(weather)

    return round(max(0.0, min(100.0, score)), 2)


def classify_weather_adjusted(row: pd.Series) -> str:
    base_class = row.get("stress_class", "unknown")
    weather_class = row.get("weather_stress_class", "")

    weather_score = row.get("weather_stress_score")
    active_alerts = row.get("weather_active_alert_count", 0)

    if pd.isna(weather_score):
        return f"{base_class} / weather unknown"

    if int(active_alerts or 0) > 0:
        return f"{base_class} / weather alert"

    if float(weather_score) >= 50:
        return f"{base_class} / high weather stress"

    if float(weather_score) >= 25:
        return f"{base_class} / moderate weather stress"

    return f"{base_class} / low weather stress"


def main() -> None:
    args = parse_args()

    fusion_csv = resolve_path(args.fusion_csv)
    weather_csv = resolve_path(args.weather_csv)
    out_path = resolve_path(args.out)
    summary_out = resolve_path(args.summary_out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    if not fusion_csv.exists():
        raise FileNotFoundError(f"Missing fusion CSV: {fusion_csv}")

    if not weather_csv.exists():
        raise FileNotFoundError(f"Missing weather CSV: {weather_csv}")

    fusion = pd.read_csv(fusion_csv)
    weather = pd.read_csv(weather_csv)

    required_fusion_cols = {
        "station_id",
        "station_name",
        "station_lat",
        "station_lon",
        "bike_unusable_score",
        "fusion_mobility_stress_score",
        "stress_class",
    }

    required_weather_cols = {
        "location_id",
        "temperature_f",
        "precipitation_probability_pct",
        "wind_speed_mph_est",
        "short_forecast",
        "active_alert_count",
        "weather_stress_score",
        "weather_stress_class",
    }

    missing_fusion = required_fusion_cols - set(fusion.columns)
    missing_weather = required_weather_cols - set(weather.columns)

    if missing_fusion:
        raise ValueError(f"Fusion CSV missing columns: {sorted(missing_fusion)}")

    if missing_weather:
        raise ValueError(f"Weather CSV missing columns: {sorted(missing_weather)}")

    # Our weather context currently has fixed monitoring points.
    # Add their approximate coordinates here so stations can inherit the nearest context point.
    weather_location_coords = {
        "manhattan_midtown": (40.7580, -73.9855),
        "brooklyn_downtown": (40.6955, -73.9925),
        "queens_jackson_heights": (40.7557, -73.8831),
        "bronx_grand_concourse": (40.8262, -73.9227),
        "staten_island_st_george": (40.6437, -74.0736),
    }

    weather_rows = []

    for _, row in weather.iterrows():
        location_id = row["location_id"]

        if location_id not in weather_location_coords:
            continue

        lat, lon = weather_location_coords[location_id]

        d = row.to_dict()
        d["weather_lat"] = lat
        d["weather_lon"] = lon
        weather_rows.append(d)

    weather_points = pd.DataFrame(weather_rows)

    if weather_points.empty:
        raise ValueError("No weather rows had known coordinates.")

    attached_rows = []

    for _, station in fusion.iterrows():
        station_lat = float(station["station_lat"])
        station_lon = float(station["station_lon"])

        best_weather = None
        best_distance_m = None

        for _, weather_row in weather_points.iterrows():
            distance_m = haversine_m(
                station_lat,
                station_lon,
                float(weather_row["weather_lat"]),
                float(weather_row["weather_lon"]),
            )

            if best_distance_m is None or distance_m < best_distance_m:
                best_distance_m = distance_m
                best_weather = weather_row

        result = station.to_dict()

        if best_weather is not None:
            result.update(
                {
                    "nearest_weather_location_id": best_weather["location_id"],
                    "nearest_weather_distance_km": round(best_distance_m / 1000.0, 2),
                    "weather_temperature_f": best_weather.get("temperature_f"),
                    "weather_precipitation_probability_pct": best_weather.get(
                        "precipitation_probability_pct"
                    ),
                    "weather_wind_speed_mph_est": best_weather.get("wind_speed_mph_est"),
                    "weather_short_forecast": best_weather.get("short_forecast"),
                    "weather_active_alert_count": int(
                        best_weather.get("active_alert_count", 0) or 0
                    ),
                    "weather_stress_score": best_weather.get("weather_stress_score"),
                    "weather_stress_class": best_weather.get("weather_stress_class"),
                }
            )
        else:
            result.update(
                {
                    "nearest_weather_location_id": None,
                    "nearest_weather_distance_km": None,
                    "weather_temperature_f": None,
                    "weather_precipitation_probability_pct": None,
                    "weather_wind_speed_mph_est": None,
                    "weather_short_forecast": None,
                    "weather_active_alert_count": None,
                    "weather_stress_score": None,
                    "weather_stress_class": None,
                }
            )

        attached_rows.append(result)

    fused_weather = pd.DataFrame(attached_rows)

    fused_weather["weather_adjusted_mobility_stress_score"] = fused_weather.apply(
        build_weather_adjusted_score,
        axis=1,
    )

    fused_weather["weather_adjusted_stress_class"] = fused_weather.apply(
        classify_weather_adjusted,
        axis=1,
    )

    fused_weather.to_csv(out_path, index=False)

    summary = (
        fused_weather.groupby(
            ["nearest_weather_location_id", "weather_stress_class"],
            dropna=False,
        )
        .agg(
            station_count=("station_id", "count"),
            avg_bike_unusable_score=("bike_unusable_score", "mean"),
            avg_local_traffic_stress_score=("local_traffic_stress_score", "mean"),
            avg_fusion_mobility_stress_score=("fusion_mobility_stress_score", "mean"),
            avg_weather_stress_score=("weather_stress_score", "mean"),
            avg_weather_adjusted_mobility_stress_score=(
                "weather_adjusted_mobility_stress_score",
                "mean",
            ),
            max_weather_active_alert_count=("weather_active_alert_count", "max"),
        )
        .reset_index()
    )

    numeric_cols = [
        "avg_bike_unusable_score",
        "avg_local_traffic_stress_score",
        "avg_fusion_mobility_stress_score",
        "avg_weather_stress_score",
        "avg_weather_adjusted_mobility_stress_score",
    ]

    summary[numeric_cols] = summary[numeric_cols].round(2)
    summary.to_csv(summary_out, index=False)

    print("Weather context attached to mobility fusion.")
    print(f"fusion_csv:  {fusion_csv}")
    print(f"weather_csv: {weather_csv}")
    print(f"out:         {out_path}")
    print(f"summary:     {summary_out}")

    print("\nSummary:")
    print(summary)

    print("\nTop weather-adjusted stress stations:")
    print(
        fused_weather.sort_values(
            "weather_adjusted_mobility_stress_score",
            ascending=False,
        )[
            [
                "station_name",
                "bike_unusable_score",
                "local_traffic_stress_score",
                "weather_stress_score",
                "weather_adjusted_mobility_stress_score",
                "nearest_weather_location_id",
                "weather_stress_class",
                "weather_adjusted_stress_class",
            ]
        ].head(20)
    )


if __name__ == "__main__":
    main()
