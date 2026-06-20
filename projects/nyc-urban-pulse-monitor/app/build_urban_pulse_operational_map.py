from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Any

import duckdb
import folium
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EARTH_RADIUS_M = 6_371_000.0

NYC_VIEW_BOUNDS = [[40.55, -74.25], [40.92, -73.68]]

NYC_LAT_MIN = 40.45
NYC_LAT_MAX = 41.00
NYC_LON_MIN = -74.35
NYC_LON_MAX = -73.45

WEATHER_LOCATION_COORDS = {
    "manhattan_midtown": (40.7580, -73.9855),
    "brooklyn_downtown": (40.6955, -73.9925),
    "queens_jackson_heights": (40.7557, -73.8831),
    "bronx_grand_concourse": (40.8262, -73.9227),
    "staten_island_st_george": (40.6437, -74.0736),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stable NYC Urban Pulse operational map.")

    parser.add_argument(
        "--fusion-weather-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "mobility_stress_station_fusion_weather.csv",
    )
    parser.add_argument(
        "--traffic-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "dot_traffic_current_segments.csv",
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "weather_current_context.csv",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "urban_pulse.duckdb",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "maps" / "urban_pulse_operational_map.html",
    )
    parser.add_argument("--window-label", type=str, default="Latest observed window")
    parser.add_argument("--traffic-stress-threshold", type=float, default=50.0)
    parser.add_argument("--min-station-score", type=float, default=0.0)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


def haversine_m(a: list[float], b: list[float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def in_nyc_bounds(point: list[float]) -> bool:
    lat, lon = point
    return NYC_LAT_MIN <= lat <= NYC_LAT_MAX and NYC_LON_MIN <= lon <= NYC_LON_MAX


def valid_polyline(points: list[list[float]]) -> bool:
    if len(points) < 2:
        return False

    if not all(in_nyc_bounds(p) for p in points):
        return False

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    if max(lats) - min(lats) > 0.25:
        return False

    if max(lons) - min(lons) > 0.35:
        return False

    # Drop impossible jumps between consecutive points.
    for a, b in zip(points[:-1], points[1:]):
        if haversine_m(a, b) > 12_000:
            return False

    return True


def parse_link_points(value: Any) -> list[list[float]]:
    if value is None or pd.isna(value):
        return []

    points: list[list[float]] = []

    for token in str(value).split():
        try:
            lat_str, lon_str = token.split(",", 1)
            points.append([float(lat_str), float(lon_str)])
        except Exception:
            continue

    return points


def color_for_score(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "#737373"

    score = max(0.0, min(100.0, float(score)))

    if score < 10:
        return "#0b8f4d"
    if score < 25:
        return "#74a832"
    if score < 50:
        return "#d6b21f"
    if score < 75:
        return "#c46a1a"
    return "#b91c1c"


def station_score(row: pd.Series) -> float | None:
    for col in [
        "weather_adjusted_mobility_stress_score",
        "fusion_mobility_stress_score",
        "bike_unusable_score",
    ]:
        if col in row.index:
            val = safe_float(row.get(col))
            if val is not None:
                return val
    return None


def station_radius(score: float | None) -> float:
    if score is None:
        return 5.0
    return max(5.0, min(17.0, 5.0 + score / 8.0))


def traffic_weight(score: float | None) -> float:
    if score is None:
        return 2.0
    return max(2.0, min(6.0, 2.0 + score / 25.0))


def load_traffic(traffic_csv: Path, db_path: Path) -> pd.DataFrame:
    traffic = pd.read_csv(traffic_csv)
    traffic["link_id"] = traffic["link_id"].astype(str)

    if "is_fresh" in traffic.columns:
        traffic = traffic[traffic["is_fresh"].map(is_truthy)].copy()

    traffic["traffic_stress_score"] = pd.to_numeric(
        traffic["traffic_stress_score"], errors="coerce"
    )

    if db_path.exists():
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            geom = con.execute(
                """
                SELECT
                    link_id,
                    link_points
                FROM dim_dot_traffic_segments
                """
            ).fetchdf()
            geom["link_id"] = geom["link_id"].astype(str)
            traffic = traffic.drop(columns=["link_points"], errors="ignore").merge(
                geom, on="link_id", how="left"
            )
        except Exception as exc:
            print(f"Warning: could not read traffic geometry from DuckDB: {exc}")

    if "link_points" not in traffic.columns:
        traffic["link_points"] = None

    polylines = []
    valid = []

    for _, row in traffic.iterrows():
        points = parse_link_points(row.get("link_points"))

        if not valid_polyline(points):
            start_lat = safe_float(row.get("start_lat"))
            start_lon = safe_float(row.get("start_lon"))
            end_lat = safe_float(row.get("end_lat"))
            end_lon = safe_float(row.get("end_lon"))

            if None not in (start_lat, start_lon, end_lat, end_lon):
                points = [[start_lat, start_lon], [end_lat, end_lon]]

        ok = valid_polyline(points)
        polylines.append(points if ok else [])
        valid.append(ok)

    traffic["polyline_points"] = polylines
    traffic["has_valid_geometry"] = valid

    before = len(traffic)
    traffic = traffic[traffic["has_valid_geometry"]].copy()
    print(f"Traffic geometry kept: {len(traffic)}/{before}")

    return traffic


def station_popup(row: pd.Series) -> folium.Popup:
    name = html.escape(str(row.get("station_name", "")))
    score = station_score(row)

    nearest_traffic = row.get("nearest_traffic_link_name")
    if pd.isna(nearest_traffic):
        nearest_traffic = "No nearby DOT detector"

    popup = f"""
    <div class="up-popup">
      <div class="up-popup-title">{name}</div>
      <div class="up-popup-sub">{html.escape(str(row.get("stress_class", "")))}</div>
      <div class="up-popup-grid">
        <div><span>Bike</span><b>{safe_float(row.get("bike_unusable_score"))}</b></div>
        <div><span>Traffic</span><b>{safe_float(row.get("local_traffic_stress_score"))}</b></div>
        <div><span>Weather</span><b>{safe_float(row.get("weather_stress_score"))}</b></div>
        <div><span>Fused</span><b>{score}</b></div>
        <div><span>Rent</span><b>{safe_float(row.get("unable_to_rent_score"))}</b></div>
        <div><span>Dock</span><b>{safe_float(row.get("unable_to_dock_score"))}</b></div>
      </div>
      <div class="up-popup-ratios">
        <div><span>Nearest traffic</span><b>{html.escape(str(nearest_traffic))}</b></div>
        <div><span>Weather point</span><b>{html.escape(str(row.get("nearest_weather_location_id", "—")))}</b></div>
        <div><span>Forecast</span><b>{html.escape(str(row.get("weather_short_forecast", "—")))}</b></div>
      </div>
    </div>
    """

    return folium.Popup(popup, max_width=360)


def traffic_popup(row: pd.Series) -> folium.Popup:
    name = html.escape(str(row.get("link_name", "")))

    popup = f"""
    <div class="up-popup">
      <div class="up-popup-title">{name}</div>
      <div class="up-popup-sub">{html.escape(str(row.get("borough", "")))}</div>
      <div class="up-popup-grid">
        <div><span>Stress</span><b>{safe_float(row.get("traffic_stress_score"))}</b></div>
        <div><span>Speed</span><b>{safe_float(row.get("current_speed_mph"))}</b></div>
        <div><span>Baseline</span><b>{safe_float(row.get("p75_speed_mph"))}</b></div>
        <div><span>Travel</span><b>{row.get("travel_time_sec", "—")}</b></div>
        <div><span>Fresh</span><b>{row.get("is_fresh", "—")}</b></div>
        <div><span>Link</span><b>{html.escape(str(row.get("link_id", "")))}</b></div>
      </div>
    </div>
    """

    return folium.Popup(popup, max_width=340)


def weather_popup(row: pd.Series) -> folium.Popup:
    location = html.escape(str(row.get("location_id", "")))

    popup = f"""
    <div class="up-popup">
      <div class="up-popup-title">{location}</div>
      <div class="up-popup-sub">{html.escape(str(row.get("weather_stress_class", "")))}</div>
      <div class="up-popup-grid">
        <div><span>Temp</span><b>{row.get("temperature_f", "—")}°F</b></div>
        <div><span>Rain</span><b>{row.get("precipitation_probability_pct", "—")}%</b></div>
        <div><span>Wind</span><b>{row.get("wind_speed_mph_est", "—")}</b></div>
        <div><span>Alerts</span><b>{row.get("active_alert_count", "—")}</b></div>
        <div><span>Stress</span><b>{row.get("weather_stress_score", "—")}</b></div>
        <div><span>State</span><b>{html.escape(str(row.get("short_forecast", "—")))}</b></div>
      </div>
    </div>
    """

    return folium.Popup(popup, max_width=330)


def add_traffic(map_obj: folium.Map, traffic: pd.DataFrame, threshold: float) -> tuple[int, int]:
    stressed = folium.FeatureGroup(
        name=f"DOT traffic · stressed links >= {threshold:.0f}", show=True
    )
    all_links = folium.FeatureGroup(name="DOT traffic · all valid links", show=False)

    stressed_count = 0

    for _, row in traffic.iterrows():
        score = safe_float(row.get("traffic_stress_score"))
        color = color_for_score(score)

        kwargs = dict(
            locations=row["polyline_points"],
            color=color,
            weight=traffic_weight(score),
            opacity=0.45,
            popup=traffic_popup(row),
            tooltip=f"{row.get('link_name', '')} · traffic stress {score}",
        )

        folium.PolyLine(**kwargs).add_to(all_links)

        if score is not None and score >= threshold:
            folium.PolyLine(**kwargs).add_to(stressed)
            stressed_count += 1

    stressed.add_to(map_obj)
    all_links.add_to(map_obj)

    return stressed_count, len(traffic)


def add_stations(map_obj: folium.Map, stations: pd.DataFrame) -> None:
    group = folium.FeatureGroup(name="Citi Bike stations · fused stress", show=True)

    for _, row in stations.iterrows():
        score = station_score(row)
        color = color_for_score(score)

        folium.CircleMarker(
            location=[float(row["station_lat"]), float(row["station_lon"])],
            radius=station_radius(score),
            color="#111111",
            fill=True,
            fill_color=color,
            fill_opacity=0.90,
            weight=1.0,
            popup=station_popup(row),
            tooltip=f"{row['station_name']} · fused stress {score}",
        ).add_to(group)

    group.add_to(map_obj)


def add_weather(map_obj: folium.Map, weather: pd.DataFrame) -> None:
    group = folium.FeatureGroup(name="NWS weather context points", show=True)

    for _, row in weather.iterrows():
        location_id = row.get("location_id")
        if location_id not in WEATHER_LOCATION_COORDS:
            continue

        lat, lon = WEATHER_LOCATION_COORDS[location_id]
        score = safe_float(row.get("weather_stress_score"))
        color = color_for_score(score)

        folium.CircleMarker(
            location=[lat, lon],
            radius=22,
            color="#111111",
            fill=True,
            fill_color=color,
            fill_opacity=0.96,
            weight=2.5,
            popup=weather_popup(row),
            tooltip=f"{location_id} · weather stress {score}",
        ).add_to(group)

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f"<div class='weather-label'>{html.escape(str(location_id).split('_')[0].upper())}</div>"
            ),
        ).add_to(group)

    group.add_to(map_obj)


def add_overlays(
    map_obj: folium.Map,
    *,
    station_count: int,
    stressed_traffic_count: int,
    all_traffic_count: int,
    weather_count: int,
    window_label: str,
) -> None:
    title = f"""
    <div class="up-title-panel">
      <div class="up-system">NYC URBAN PULSE</div>
      <div class="up-product">OPERATIONAL MAP</div>
      <div class="up-rule"></div>
      <div class="up-line">{html.escape(window_label).upper()}</div>
      <div class="up-line">{station_count:,} STATIONS · {stressed_traffic_count}/{all_traffic_count} STRESSED TRAFFIC LINKS · {weather_count} WEATHER POINTS</div>
      <div class="up-line">BIKE · TRAFFIC · WEATHER</div>
    </div>
    """

    legend = """
    <div class="up-legend">
      <div class="up-legend-head">STRESS SCORE</div>
      <div class="up-legend-row"><span style="background:#0b8f4d"></span>0–10 LOW</div>
      <div class="up-legend-row"><span style="background:#74a832"></span>10–25 ELEVATED</div>
      <div class="up-legend-row"><span style="background:#d6b21f"></span>25–50 MIXED</div>
      <div class="up-legend-row"><span style="background:#c46a1a"></span>50–75 HIGH</div>
      <div class="up-legend-row"><span style="background:#b91c1c"></span>75–100 CRITICAL</div>
      <div class="up-legend-foot">CIRCLES = STATIONS / WEATHER · LINES = TRAFFIC</div>
    </div>
    """

    css = """
    <style>
      .leaflet-container {
        font-family: "Courier New", Courier, monospace !important;
      }

      .up-title-panel {
        position: fixed;
        top: 26px;
        left: 28px;
        z-index: 9999;
        width: 430px;
        color: #111;
        font-family: "Courier New", Courier, monospace;
        pointer-events: none;
      }

      .up-title-panel::before {
        content: "";
        display: block;
        width: 22px;
        height: 5px;
        background: #f2c230;
        margin-bottom: 13px;
      }

      .up-system {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.19em;
        color: #646464;
        margin-bottom: 7px;
      }

      .up-product {
        font-size: 27px;
        line-height: 0.95;
        font-weight: 900;
        letter-spacing: -0.05em;
      }

      .up-rule {
        width: 100%;
        height: 1px;
        background: rgba(17, 17, 17, 0.72);
        margin: 13px 0 10px 0;
      }

      .up-line {
        font-size: 11px;
        line-height: 1.65;
        font-weight: 700;
      }

      .up-legend {
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        width: 255px;
        color: #111;
        font-family: "Courier New", Courier, monospace;
        pointer-events: none;
      }

      .up-legend::before {
        content: "";
        display: block;
        width: 100%;
        height: 1px;
        background: rgba(17, 17, 17, 0.72);
        margin-bottom: 10px;
      }

      .up-legend-head {
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 0.18em;
        margin-bottom: 9px;
      }

      .up-legend-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        font-weight: 700;
        line-height: 1.65;
      }

      .up-legend-row span {
        width: 10px;
        height: 10px;
        display: inline-block;
        border-radius: 50%;
        border: 1px solid rgba(0,0,0,0.15);
      }

      .up-legend-foot {
        margin-top: 9px;
        padding-top: 8px;
        border-top: 1px solid rgba(17, 17, 17, 0.5);
        color: #646464;
        font-size: 10px;
        font-weight: 700;
      }

      .weather-label {
        margin-left: 20px;
        margin-top: -10px;
        padding: 2px 5px;
        background: rgba(250,250,247,0.92);
        border-left: 2px solid #111;
        font-family: "Courier New", Courier, monospace;
        font-size: 10px;
        font-weight: 900;
        color: #111;
        white-space: nowrap;
      }

      .leaflet-control-layers {
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
        font-family: "Courier New", Courier, monospace !important;
      }

      .leaflet-control-layers-expanded {
        background: rgba(250,250,247,0.92) !important;
        border-left: 1px solid rgba(17,17,17,0.7) !important;
        padding: 10px 12px !important;
      }

      .leaflet-popup-content-wrapper {
        border-radius: 0 !important;
        background: rgba(250, 250, 247, 0.98) !important;
        box-shadow: 8px 8px 0 rgba(17, 17, 17, 0.16) !important;
        border: 1px solid rgba(17, 17, 17, 0.62) !important;
      }

      .leaflet-popup-content {
        margin: 14px !important;
      }

      .up-popup {
        width: 315px;
        font-family: "Courier New", Courier, monospace;
        color: #111;
      }

      .up-popup-title {
        font-size: 15px;
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -0.04em;
        margin-bottom: 4px;
        text-transform: uppercase;
      }

      .up-popup-sub {
        font-size: 10px;
        font-weight: 700;
        color: #666;
        margin-bottom: 11px;
        text-transform: uppercase;
      }

      .up-popup-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        border-top: 1px solid #222;
        border-left: 1px solid #222;
        margin-bottom: 11px;
      }

      .up-popup-grid div {
        padding: 7px 6px;
        border-right: 1px solid #222;
        border-bottom: 1px solid #222;
      }

      .up-popup-grid span {
        display: block;
        font-size: 9px;
        color: #666;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 3px;
      }

      .up-popup-grid b {
        font-size: 13px;
        font-weight: 900;
      }

      .up-popup-ratios {
        border-top: 1px solid #222;
        padding-top: 8px;
      }

      .up-popup-ratios div {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        font-size: 10px;
        line-height: 1.8;
        font-weight: 700;
      }

      .up-popup-ratios span {
        color: #555;
        text-transform: uppercase;
      }

      .up-popup-ratios b {
        max-width: 180px;
        text-align: right;
        font-weight: 900;
      }
    </style>
    """

    map_obj.get_root().header.add_child(folium.Element(css))
    map_obj.get_root().html.add_child(folium.Element(title))
    map_obj.get_root().html.add_child(folium.Element(legend))


def main() -> None:
    args = parse_args()

    fusion_weather_csv = resolve_path(args.fusion_weather_csv)
    traffic_csv = resolve_path(args.traffic_csv)
    weather_csv = resolve_path(args.weather_csv)
    db_path = resolve_path(args.db)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(fusion_weather_csv)
    weather = pd.read_csv(weather_csv)
    traffic = load_traffic(traffic_csv, db_path)

    stations = stations.dropna(subset=["station_lat", "station_lon"]).copy()
    stations["map_score"] = stations.apply(station_score, axis=1)
    stations = stations[stations["map_score"].fillna(0) >= args.min_station_score].copy()

    print("Station bounds:")
    print(stations[["station_lat", "station_lon"]].describe())

    map_obj = folium.Map(
        location=[40.73, -73.98],
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )

    stressed_count, all_traffic_count = add_traffic(
        map_obj,
        traffic,
        args.traffic_stress_threshold,
    )
    add_stations(map_obj, stations)
    add_weather(map_obj, weather)

    map_obj.fit_bounds(NYC_VIEW_BOUNDS, padding=(20, 20))

    folium.LayerControl(collapsed=False, position="topright").add_to(map_obj)

    add_overlays(
        map_obj,
        station_count=len(stations),
        stressed_traffic_count=stressed_count,
        all_traffic_count=all_traffic_count,
        weather_count=len(weather),
        window_label=args.window_label,
    )

    map_obj.save(out_path)

    print("Operational map built.")
    print(f"stations:               {len(stations)}")
    print(f"traffic links total:    {all_traffic_count}")
    print(f"traffic links stressed: {stressed_count}")
    print(f"weather points:         {len(weather)}")
    print(f"out:                    {out_path}")


if __name__ == "__main__":
    main()
