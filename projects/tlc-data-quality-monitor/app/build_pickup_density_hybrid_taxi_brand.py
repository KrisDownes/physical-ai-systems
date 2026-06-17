from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hybrid animated deck.gl showpiece map: filled taxi-zone polygons, "
            "subtle extrusion, glow-like outlines, full-hour animation, and compact top-area annotation."
        )
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
        "--zones",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "tlc" / "taxi_zones_4326.parquet",
        help="Taxi zone geometry file. Supports .parquet, .geojson, or .json.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path.",
    )

    parser.add_argument(
        "--max-elevation",
        type=float,
        default=1800.0,
        help="Maximum polygon extrusion height in meters for the busiest zone/hour.",
    )

    parser.add_argument(
        "--start-hour",
        type=int,
        default=6,
        help="Initial hour shown, 0-23.",
    )

    parser.add_argument(
        "--ms-per-hour",
        type=int,
        default=650,
        help="Animation speed in milliseconds per hour. Lower is faster.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_zones(path: Path) -> gpd.GeoDataFrame:
    """Read taxi zone geometry from GeoJSON/JSON or GeoParquet."""
    suffix = path.suffix.lower()

    if suffix in {".geojson", ".json"}:
        return gpd.read_file(path)

    if suffix == ".parquet":
        return gpd.read_parquet(path)

    raise ValueError(f"Unsupported zone geometry file type: {path}")


def find_location_id_column(gdf: gpd.GeoDataFrame) -> str:
    cols_lower = {c.lower(): c for c in gdf.columns}

    if "locationid" in cols_lower:
        return cols_lower["locationid"]

    if "location_id" in cols_lower:
        return cols_lower["location_id"]

    raise ValueError(
        "Could not find LocationID/location_id in taxi zone geometry columns: "
        f"{list(gdf.columns)}"
    )


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure geometries are in lon/lat coordinates."""
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326")

    if gdf.crs.to_epsg() != 4326:
        return gdf.to_crs("EPSG:4326")

    return gdf


def load_zone_geojson(zones_path: Path) -> dict:
    """
    Load taxi zone polygons as GeoJSON.

    The modern TLC trip data gives pickup/dropoff LocationID values, so this
    map fills and extrudes the full taxi-zone polygons rather than exact pickup
    points.
    """
    if not zones_path.exists():
        raise FileNotFoundError(f"Missing taxi zone geometry file: {zones_path}")

    gdf = read_zones(zones_path)
    location_col = find_location_id_column(gdf)
    gdf = ensure_wgs84(gdf).copy()

    gdf["location_id"] = gdf[location_col].astype(int)

    if "zone" not in gdf.columns and "Zone" in gdf.columns:
        gdf["zone"] = gdf["Zone"].astype(str)
    elif "zone" not in gdf.columns:
        gdf["zone"] = ""

    if "borough" not in gdf.columns and "Borough" in gdf.columns:
        gdf["borough"] = gdf["Borough"].astype(str)
    elif "borough" not in gdf.columns:
        gdf["borough"] = ""

    if "service_zone" not in gdf.columns and "servicezone" in gdf.columns:
        gdf["service_zone"] = gdf["servicezone"].astype(str)
    elif "service_zone" not in gdf.columns:
        gdf["service_zone"] = ""

    keep_cols = ["location_id", "zone", "borough", "service_zone", "geometry"]

    return json.loads(gdf[keep_cols].to_json())


def load_pickup_counts(
    db_path: Path,
    taxi_type: str,
    year: int,
    month: int,
) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Missing DuckDB file: {db_path}")

    con = duckdb.connect(str(db_path))

    return con.execute(
        """
        SELECT
            pickup_location_id AS location_id,
            pickup_hour,
            COUNT(*) AS trip_count,
            SUM(total_amount) AS gross_total_amount,
            AVG(total_amount) AS avg_total_amount
        FROM clean_yellow_trips
        WHERE
            taxi_type = ?
            AND data_year = ?
            AND data_month = ?
        GROUP BY
            pickup_location_id,
            pickup_hour
        ORDER BY
            pickup_hour,
            trip_count DESC
        """,
        [taxi_type, year, month],
    ).fetchdf()


def attach_hourly_series_to_zones(
    zones_geojson: dict,
    counts_df: pd.DataFrame,
) -> tuple[dict, int]:
    """
    Add 24-hour pickup-count series to each zone feature's properties.

    Output is a GeoJSON FeatureCollection where every feature has:
    - series: 24 pickup counts
    - gross_series: 24 gross total amounts
    """
    counts_by_zone_hour: dict[int, dict[int, dict]] = {}

    for row in counts_df.itertuples(index=False):
        location_id = int(row.location_id)
        hour = int(row.pickup_hour)

        if location_id not in counts_by_zone_hour:
            counts_by_zone_hour[location_id] = {}

        counts_by_zone_hour[location_id][hour] = {
            "trip_count": int(row.trip_count),
            "gross_total_amount": float(row.gross_total_amount or 0.0),
            "avg_total_amount": float(row.avg_total_amount or 0.0),
        }

    max_trip_count = int(counts_df["trip_count"].max()) if len(counts_df) > 0 else 0

    enriched = {
        "type": "FeatureCollection",
        "features": [],
    }

    for feature in zones_geojson["features"]:
        props = dict(feature.get("properties", {}))
        location_id = int(props["location_id"])

        series = [0] * 24
        gross_series = [0.0] * 24

        for hour in range(24):
            values = counts_by_zone_hour.get(location_id, {}).get(hour)
            if values is None:
                continue

            series[hour] = values["trip_count"]
            gross_series[hour] = values["gross_total_amount"]

        props["series"] = series
        props["gross_series"] = gross_series

        enriched["features"].append(
            {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": props,
            }
        )

    return enriched, max_trip_count


def write_hybrid_html(
    output_path: Path,
    zones_geojson: dict,
    taxi_type: str,
    year: int,
    month: int,
    max_trip_count: int,
    max_elevation: float,
    start_hour: int,
    ms_per_hour: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zones_json = json.dumps(zones_geojson, separators=(",", ":"))

    title = "NYC TAXI PICKUPS"
    subtitle = f"{year}-{month:02d} · pickup density by TLC taxi zone"

    html_template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>__TITLE__</title>
  <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
  <style>
    :root {
      --taxi-yellow: #f7c531;
      --taxi-yellow-deep: #e6ad10;
      --taxi-black: #0b0c0e;
      --taxi-panel: rgba(247, 197, 49, 0.94);
      --taxi-panel-soft: rgba(247, 197, 49, 0.86);
      --taxi-font: "Arial Black", "Helvetica Neue Condensed Black", "HelveticaNeue-CondensedBlack",
                   "Roboto Condensed", "Arial Narrow", Impact, sans-serif;
      --body-font: "Arial Narrow", "Roboto Condensed", Arial, sans-serif;
    }

    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #030407;
      color: #f5f7fb;
      font-family: var(--body-font);
    }

    #deck-container {
      position: absolute;
      inset: 0;
      width: 100vw;
      height: 100vh;
      background:
        radial-gradient(circle at 48% 42%, rgba(22, 30, 44, 0.55), rgba(3, 4, 7, 0.98) 68%),
        #030407;
    }

    #title-block {
      position: absolute;
      top: 24px;
      left: 28px;
      z-index: 10;
      pointer-events: none;
      background: var(--taxi-panel);
      color: var(--taxi-black);
      border: 2px solid rgba(0, 0, 0, 0.72);
      border-radius: 4px;
      padding: 12px 18px 10px 18px;
      box-shadow:
        0 10px 28px rgba(0, 0, 0, 0.36),
        inset 0 -2px 0 rgba(0, 0, 0, 0.16);
    }

    #title-block h1 {
      margin: 0;
      font-family: var(--taxi-font);
      font-size: 34px;
      font-weight: 900;
      letter-spacing: -1.2px;
      line-height: 0.96;
      text-transform: uppercase;
      color: var(--taxi-black);
    }

    #title-block .subtitle {
      margin-top: 7px;
      font-family: var(--body-font);
      font-size: 13px;
      font-weight: 700;
      color: rgba(10, 12, 14, 0.82);
      letter-spacing: 0.1px;
      text-transform: uppercase;
    }

    #hour-block {
      position: absolute;
      top: 24px;
      right: 30px;
      z-index: 10;
      text-align: right;
      pointer-events: none;
      background: var(--taxi-panel);
      color: var(--taxi-black);
      border: 2px solid rgba(0, 0, 0, 0.72);
      border-radius: 4px;
      padding: 10px 14px 9px 16px;
      min-width: 168px;
      box-shadow:
        0 10px 28px rgba(0, 0, 0, 0.36),
        inset 0 -2px 0 rgba(0, 0, 0, 0.16);
    }

    #hour-label {
      font-family: var(--taxi-font);
      font-size: 50px;
      font-weight: 900;
      letter-spacing: -1.5px;
      color: var(--taxi-black);
      line-height: 0.9;
      font-variant-numeric: tabular-nums;
    }

    #hour-caption {
      margin-top: 6px;
      font-family: var(--body-font);
      font-size: 11px;
      font-weight: 800;
      color: rgba(10, 12, 14, 0.78);
      text-transform: uppercase;
      letter-spacing: 1.2px;
    }

    #peak-block {
      position: absolute;
      top: 108px;
      right: 30px;
      z-index: 10;
      text-align: right;
      pointer-events: none;
      max-width: 360px;
      background: rgba(10, 12, 14, 0.78);
      border-left: 6px solid var(--taxi-yellow);
      padding: 11px 13px 11px 16px;
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.30);
    }

    #peak-kicker {
      font-family: var(--body-font);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 1.3px;
      color: var(--taxi-yellow);
      margin-bottom: 5px;
    }

    #peak-zone {
      font-family: var(--taxi-font);
      font-size: 24px;
      font-weight: 900;
      letter-spacing: -0.7px;
      color: rgba(255, 255, 255, 0.94);
      text-transform: uppercase;
      line-height: 1.0;
    }

    #peak-detail {
      margin-top: 5px;
      font-family: var(--body-font);
      font-size: 13px;
      font-weight: 700;
      color: rgba(235, 241, 255, 0.68);
      font-variant-numeric: tabular-nums;
    }

    #controls {
      position: absolute;
      left: 50%;
      bottom: 26px;
      transform: translateX(-50%);
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 14px;
      background: rgba(247, 197, 49, 0.92);
      border: 2px solid rgba(0, 0, 0, 0.72);
      border-radius: 4px;
      padding: 9px 16px;
      box-shadow:
        0 10px 28px rgba(0, 0, 0, 0.34),
        inset 0 -2px 0 rgba(0, 0, 0, 0.16);
    }

    #playButton {
      background: var(--taxi-black);
      color: var(--taxi-yellow);
      border: 1px solid rgba(0, 0, 0, 0.85);
      border-radius: 3px;
      padding: 7px 14px;
      font-size: 13px;
      font-weight: 900;
      cursor: pointer;
      font-family: var(--taxi-font);
      text-transform: uppercase;
      letter-spacing: -0.4px;
    }

    #playButton:hover {
      background: #202124;
    }

    #timeline {
      width: 440px;
      accent-color: var(--taxi-black);
    }

    #legend {
      position: absolute;
      left: 30px;
      bottom: 30px;
      z-index: 10;
      width: 270px;
      pointer-events: none;
      color: rgba(235, 241, 255, 0.76);
      text-shadow: 0 2px 16px rgba(0, 0, 0, 0.90);
    }

    #legend-title {
      font-family: var(--taxi-font);
      font-size: 13px;
      letter-spacing: -0.2px;
      text-transform: uppercase;
      margin-bottom: 8px;
      color: var(--taxi-yellow);
    }

    #legend-gradient {
      height: 12px;
      border-radius: 999px;
      background: linear-gradient(
        90deg,
        rgb(18, 59, 180) 0%,
        rgb(49, 135, 255) 28%,
        rgb(210, 32, 70) 70%,
        rgb(255, 144, 34) 100%
      );
      box-shadow:
        0 0 20px rgba(60, 140, 255, 0.22),
        0 0 26px rgba(255, 120, 40, 0.16);
    }

    #legend-row {
      margin-top: 6px;
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: rgba(235, 241, 255, 0.46);
    }

    #credit {
      position: absolute;
      right: 32px;
      bottom: 28px;
      z-index: 10;
      color: rgba(247, 197, 49, 0.48);
      font-family: var(--body-font);
      font-weight: 700;
      font-size: 12px;
      pointer-events: none;
      text-shadow: 0 2px 14px rgba(0, 0, 0, 0.90);
    }
  </style>
</head>
<body>
  <div id="deck-container"></div>

  <div id="title-block">
    <h1>__TITLE__</h1>
    <div class="subtitle">__SUBTITLE__</div>
  </div>

  <div id="hour-block">
    <div id="hour-label">00:00</div>
    <div id="hour-caption">hour of day</div>
  </div>

  <div id="peak-block">
    <div id="peak-kicker">top area</div>
    <div id="peak-zone">—</div>
    <div id="peak-detail"></div>
  </div>

  <div id="legend">
    <div id="legend-title">pickup density</div>
    <div id="legend-gradient"></div>
    <div id="legend-row">
      <span>lower</span>
      <span>higher</span>
    </div>
  </div>

  <div id="controls">
    <button id="playButton">Pause</button>
    <input id="timeline" type="range" min="0" max="23" step="1" value="__START_HOUR__" />
  </div>

  <div id="credit">NYC TLC trip records · taxi zone polygons</div>

  <script>
    const rawZonesGeojson = __ZONES_JSON__;
    const maxTripCount = __MAX_TRIP_COUNT__;
    const maxElevation = __MAX_ELEVATION__;
    const msPerHour = __MS_PER_HOUR__;

    const hourLabel = document.getElementById("hour-label");
    const playButton = document.getElementById("playButton");
    const timeline = document.getElementById("timeline");
    const peakZone = document.getElementById("peak-zone");
    const peakDetail = document.getElementById("peak-detail");

    let currentHour = __START_HOUR__;
    let playing = true;
    let timer = null;

    const initialViewState = {
      longitude: -73.94,
      latitude: 40.725,
      zoom: 10.35,
      pitch: 54,
      bearing: -24
    };

    function clamp(x, lo, hi) {
      return Math.max(lo, Math.min(hi, x));
    }

    function colorForValue(v, vmax, alpha = 210) {
      if (vmax <= 0) return [15, 45, 125, alpha];

      // Gamma brightens mid-range values so the map feels more filled in.
      const x = Math.pow(clamp(v / vmax, 0, 1), 0.46);

      if (x < 0.50) {
        const t = x / 0.50;
        const r = Math.round(18 + t * 50);
        const g = Math.round(59 + t * 86);
        const b = Math.round(180 + t * 75);
        return [r, g, b, alpha];
      }

      if (x < 0.82) {
        const t = (x - 0.50) / 0.32;
        const r = Math.round(68 + t * 152);
        const g = Math.round(145 - t * 112);
        const b = Math.round(255 - t * 188);
        return [r, g, b, alpha + 12];
      }

      const t = (x - 0.82) / 0.18;
      const r = Math.round(220 + t * 35);
      const g = Math.round(33 + t * 112);
      const b = Math.round(67 - t * 37);
      return [r, g, b, alpha + 25];
    }

    function lineColorForValue(v, vmax) {
      const c = colorForValue(v, vmax, 170);
      return [c[0], c[1], c[2], 185];
    }

    function elevationForValue(v, vmax) {
      if (vmax <= 0 || v <= 0) return 0;
      const x = Math.pow(clamp(v / vmax, 0, 1), 0.62);
      return x * maxElevation;
    }

    function formatHour(hour) {
      return String(hour).padStart(2, "0") + ":00";
    }

    function buildHourGeojson(hour) {
      const features = rawZonesGeojson.features.map((feature) => {
        const props = feature.properties || {};
        const count = (props.series && props.series[hour]) ? props.series[hour] : 0;
        const gross = (props.gross_series && props.gross_series[hour]) ? props.gross_series[hour] : 0;

        return {
          type: "Feature",
          geometry: feature.geometry,
          properties: {
            ...props,
            current_hour: hour,
            current_count: count,
            current_gross: gross,
            fill_color: colorForValue(count, maxTripCount, count > 0 ? 180 : 38),
            line_color: lineColorForValue(count, maxTripCount),
            elevation: elevationForValue(count, maxTripCount)
          }
        };
      });

      return {
        type: "FeatureCollection",
        features
      };
    }

    function getTopZone(hour) {
      let best = null;

      for (const feature of rawZonesGeojson.features) {
        const props = feature.properties || {};
        const count = (props.series && props.series[hour]) ? props.series[hour] : 0;

        if (!best || count > best.count) {
          best = {
            zone: props.zone || "Unknown",
            borough: props.borough || "Unknown",
            count
          };
        }
      }

      return best;
    }

    function makeLayers(hour) {
      const hourGeojson = buildHourGeojson(hour);

      const baseFillLayer = new deck.GeoJsonLayer({
        id: "zone-fill-layer",
        data: hourGeojson,
        filled: true,
        stroked: false,
        extruded: true,
        wireframe: false,
        getFillColor: f => f.properties.fill_color,
        getElevation: f => f.properties.elevation,
        material: {
          ambient: 0.42,
          diffuse: 0.58,
          shininess: 28,
          specularColor: [255, 210, 160]
        },
        pickable: true,
        autoHighlight: true,
        transitions: {
          getFillColor: 420,
          getElevation: 420
        }
      });

      const outlineGlowLayer = new deck.GeoJsonLayer({
        id: "zone-outline-glow-layer",
        data: hourGeojson,
        filled: false,
        stroked: true,
        getLineColor: f => f.properties.line_color,
        lineWidthMinPixels: 1.3,
        pickable: false,
        transitions: {
          getLineColor: 420
        }
      });

      const faintGridLayer = new deck.GeoJsonLayer({
        id: "zone-faint-grid-layer",
        data: rawZonesGeojson,
        filled: false,
        stroked: true,
        getLineColor: [215, 230, 250, 34],
        lineWidthMinPixels: 0.6,
        pickable: false
      });

      return [faintGridLayer, baseFillLayer, outlineGlowLayer];
    }

    const deckgl = new deck.DeckGL({
      container: "deck-container",
      initialViewState,
      controller: true,
      layers: makeLayers(currentHour),
      getTooltip: info => {
        if (!info.object || !info.object.properties) return null;

        const p = info.object.properties;

        return {
          text:
            `${p.zone || "Unknown"} (${p.borough || "Unknown"})\n` +
            `Hour: ${formatHour(p.current_hour)}\n` +
            `Pickups: ${Number(p.current_count || 0).toLocaleString()}\n` +
            `Gross: $${Math.round(Number(p.current_gross || 0)).toLocaleString()}`
        };
      }
    });

    function updatePeak(hour) {
      const best = getTopZone(hour);

      if (!best) {
        peakZone.textContent = "—";
        peakDetail.textContent = "";
        return;
      }

      peakZone.textContent = best.zone;
      peakDetail.textContent = `${best.borough} · ${Number(best.count).toLocaleString()} pickups`;
    }

    function renderHour(hour) {
      currentHour = Number(hour) % 24;
      hourLabel.textContent = formatHour(currentHour);
      timeline.value = currentHour;

      deckgl.setProps({
        layers: makeLayers(currentHour)
      });

      updatePeak(currentHour);
    }

    function nextHour() {
      renderHour((currentHour + 1) % 24);
    }

    function startPlayback() {
      if (timer !== null) clearInterval(timer);
      playing = true;
      playButton.textContent = "Pause";
      timer = setInterval(nextHour, msPerHour);
    }

    function stopPlayback() {
      playing = false;
      playButton.textContent = "Play";
      if (timer !== null) clearInterval(timer);
      timer = null;
    }

    playButton.addEventListener("click", () => {
      if (playing) {
        stopPlayback();
      } else {
        startPlayback();
      }
    });

    timeline.addEventListener("input", (event) => {
      stopPlayback();
      renderHour(Number(event.target.value));
    });

    renderHour(currentHour);
    startPlayback();
  </script>
</body>
</html>
"""

    html = (
        html_template
        .replace("__TITLE__", title)
        .replace("__SUBTITLE__", subtitle)
        .replace("__ZONES_JSON__", zones_json)
        .replace("__MAX_TRIP_COUNT__", str(max_trip_count))
        .replace("__MAX_ELEVATION__", str(max_elevation))
        .replace("__START_HOUR__", str(start_hour))
        .replace("__MS_PER_HOUR__", str(ms_per_hour))
    )

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.start_hour < 0 or args.start_hour > 23:
        raise ValueError("--start-hour must be between 0 and 23.")

    db_path = resolve_path(args.db)
    zones_path = resolve_path(args.zones)

    if args.output is None:
        output_path = (
            PROJECT_ROOT
            / "outputs"
            / "reports"
            / "maps"
            / f"{args.taxi_type}_{args.year}_{args.month:02d}_pickup_density_hybrid.html"
        )
    else:
        output_path = resolve_path(args.output)

    zones_geojson = load_zone_geojson(zones_path)
    counts_df = load_pickup_counts(
        db_path=db_path,
        taxi_type=args.taxi_type,
        year=args.year,
        month=args.month,
    )

    enriched_zones_geojson, max_trip_count = attach_hourly_series_to_zones(
        zones_geojson=zones_geojson,
        counts_df=counts_df,
    )

    write_hybrid_html(
        output_path=output_path,
        zones_geojson=enriched_zones_geojson,
        taxi_type=args.taxi_type,
        year=args.year,
        month=args.month,
        max_trip_count=max_trip_count,
        max_elevation=args.max_elevation,
        start_hour=args.start_hour,
        ms_per_hour=args.ms_per_hour,
    )

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
