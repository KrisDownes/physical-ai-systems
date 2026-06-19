from __future__ import annotations

import argparse
import html
from pathlib import Path

import folium
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


LAYER_CONFIG = {
    "overall": {
        "label": "Overall unusable",
        "metric": "any_failure_ratio",
        "minutes": ["empty_minutes", "full_minutes", "offline_minutes"],
        "tooltip": "Unusable",
    },
    "rent": {
        "label": "Unable to rent",
        "metric": "rental_failure_ratio",
        "minutes": ["empty_minutes", "offline_minutes"],
        "tooltip": "Unable to rent",
    },
    "dock": {
        "label": "Unable to dock",
        "metric": "return_failure_ratio",
        "minutes": ["full_minutes", "offline_minutes"],
        "tooltip": "Unable to dock",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a minimalist Citi Bike reliability map."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "reports"
        / "citibike_station_reliability_summary.csv",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "maps"
        / "citibike_station_reliability_map.html",
    )

    parser.add_argument(
        "--min-observed-min",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--window-label",
        type=str,
        default="Latest observed hour",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="If greater than zero, show only the top N stations by overall unusable ratio.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def pct(value: float) -> str:
    return f"{100.0 * float(value):.0f}%"


def color_for_ratio(value: float) -> str:
    value = max(0.0, min(1.0, float(value)))

    if value < 0.10:
        return "#0b8f4d"
    if value < 0.25:
        return "#74a832"
    if value < 0.50:
        return "#d6b21f"
    if value < 0.75:
        return "#c46a1a"
    return "#b91c1c"


def minutes_for_layer(row: pd.Series, layer_key: str) -> float:
    return sum(float(row.get(col, 0.0)) for col in LAYER_CONFIG[layer_key]["minutes"])


def marker_radius(row: pd.Series, layer_key: str) -> float:
    minutes = minutes_for_layer(row, layer_key)
    return max(4.0, min(18.0, 4.0 + minutes / 7.0))


def make_popup(row: pd.Series) -> folium.Popup:
    station_name = html.escape(str(row["station_name"]))

    unusable_minutes = (
        float(row["empty_minutes"])
        + float(row["full_minutes"])
        + float(row["offline_minutes"])
    )

    popup_html = f"""
    <div class="up-popup">
        <div class="up-popup-title">{station_name}</div>
        <div class="up-popup-sub">
            {row["observed_minutes"]:.0f} min observed · {int(row["capacity"])} docks
        </div>

        <div class="up-popup-grid">
            <div>
                <span>Available</span>
                <b>{row["available_minutes"]:.0f}</b>
            </div>
            <div>
                <span>Unusable</span>
                <b>{unusable_minutes:.0f}</b>
            </div>
            <div>
                <span>Empty</span>
                <b>{row["empty_minutes"]:.0f}</b>
            </div>
            <div>
                <span>Full</span>
                <b>{row["full_minutes"]:.0f}</b>
            </div>
            <div>
                <span>Offline</span>
                <b>{row["offline_minutes"]:.0f}</b>
            </div>
            <div>
                <span>Station ID</span>
                <b>{html.escape(str(row["station_id"]))[:8]}</b>
            </div>
        </div>

        <div class="up-popup-ratios">
            <div><span>Unable to rent</span><b>{pct(row["rental_failure_ratio"])}</b></div>
            <div><span>Unable to dock</span><b>{pct(row["return_failure_ratio"])}</b></div>
            <div><span>Overall unusable</span><b>{pct(row["any_failure_ratio"])}</b></div>
        </div>
    </div>
    """

    return folium.Popup(popup_html, max_width=330)


def add_layer(map_obj: folium.Map, df: pd.DataFrame, layer_key: str, *, show: bool) -> None:
    config = LAYER_CONFIG[layer_key]
    group = folium.FeatureGroup(name=config["label"], show=show)

    metric = config["metric"]

    for _, row in df.iterrows():
        value = float(row[metric])
        color = color_for_ratio(value)

        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=marker_radius(row, layer_key),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.76,
            weight=1.15,
            popup=make_popup(row),
            tooltip=f"{row['station_name']} · {config['tooltip']}: {pct(value)}",
        ).add_to(group)

    group.add_to(map_obj)


def add_title(map_obj: folium.Map, window_label: str, station_count: int, observed_minutes: float) -> None:
    title_html = f"""
    <div class="up-title-panel">
        <div class="up-system">NYC URBAN PULSE</div>
        <div class="up-product">CITI BIKE RELIABILITY</div>
        <div class="up-rule"></div>
        <div class="up-line">{html.escape(window_label).upper()}</div>
        <div class="up-line">OBSERVED {observed_minutes:.0f} MIN · {station_count:,} STATIONS</div>
        <div class="up-line">UNUSABLE = EMPTY · FULL · OFFLINE</div>
    </div>
    """

    map_obj.get_root().html.add_child(folium.Element(title_html))


def add_legend(map_obj: folium.Map) -> None:
    legend_html = """
    <div class="up-legend">
        <div class="up-legend-head">STATUS</div>
        <div class="up-legend-row"><span style="background:#0b8f4d"></span>USABLE</div>
        <div class="up-legend-row"><span style="background:#74a832"></span>LOW ISSUE</div>
        <div class="up-legend-row"><span style="background:#d6b21f"></span>MIXED</div>
        <div class="up-legend-row"><span style="background:#c46a1a"></span>POOR</div>
        <div class="up-legend-row"><span style="background:#b91c1c"></span>CRITICAL</div>
        <div class="up-legend-foot">SIZE = MINUTES UNUSABLE</div>
    </div>
    """

    map_obj.get_root().html.add_child(folium.Element(legend_html))


def add_css(map_obj: folium.Map) -> None:
    css = """
    <style>
        :root {
            --ink: #111111;
            --muted: #646464;
            --paper: rgba(250, 250, 247, 0.94);
            --line: rgba(17, 17, 17, 0.72);
            --yellow: #f2c230;
        }

        .leaflet-container {
            background: #f5f5f0 !important;
            font-family: "Courier New", Courier, monospace !important;
        }

        .up-title-panel {
            position: fixed;
            top: 26px;
            left: 28px;
            z-index: 9999;
            width: 330px;
            padding: 0;
            color: var(--ink);
            font-family: "Courier New", Courier, monospace;
            letter-spacing: 0.01em;
            pointer-events: none;
        }

        .up-title-panel::before {
            content: "";
            display: block;
            width: 22px;
            height: 5px;
            background: var(--yellow);
            margin-bottom: 13px;
        }

        .up-system {
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.19em;
            color: var(--muted);
            margin-bottom: 7px;
        }

        .up-product {
            font-size: 25px;
            line-height: 0.95;
            font-weight: 900;
            letter-spacing: -0.04em;
            color: #0b0b0b;
        }

        .up-rule {
            width: 100%;
            height: 1px;
            background: var(--line);
            margin: 13px 0 10px 0;
        }

        .up-line {
            font-size: 11px;
            line-height: 1.65;
            font-weight: 700;
            color: #222;
        }

        .up-legend {
            position: fixed;
            bottom: 30px;
            left: 30px;
            z-index: 9999;
            width: 190px;
            color: var(--ink);
            font-family: "Courier New", Courier, monospace;
            pointer-events: none;
        }

        .up-legend::before {
            content: "";
            display: block;
            width: 100%;
            height: 1px;
            background: var(--line);
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
            color: var(--muted);
            font-size: 10px;
            font-weight: 700;
        }

        .leaflet-control-layers {
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
            font-family: "Courier New", Courier, monospace !important;
            margin-top: 26px !important;
            margin-right: 28px !important;
        }

        .leaflet-control-layers-expanded {
            background: rgba(250, 250, 247, 0.82) !important;
            border-left: 1px solid rgba(17,17,17,0.7) !important;
            padding: 10px 12px !important;
            backdrop-filter: blur(2px);
        }

        .leaflet-control-layers label {
            font-size: 11px !important;
            font-weight: 800 !important;
            color: #111 !important;
            letter-spacing: 0.03em;
            margin: 6px 0 !important;
        }

        .leaflet-control-layers-selector {
            accent-color: #111111;
        }

        .leaflet-control-zoom {
            border: none !important;
            box-shadow: none !important;
        }

        .leaflet-control-zoom a {
            border-radius: 0 !important;
            background: rgba(250, 250, 247, 0.9) !important;
            color: #111 !important;
            border: 1px solid rgba(17,17,17,0.25) !important;
            font-family: "Courier New", Courier, monospace !important;
        }

        .leaflet-popup-content-wrapper {
            border-radius: 0 !important;
            background: rgba(250, 250, 247, 0.97) !important;
            box-shadow: 8px 8px 0 rgba(17, 17, 17, 0.16) !important;
            border: 1px solid rgba(17, 17, 17, 0.62) !important;
        }

        .leaflet-popup-tip {
            background: rgba(250, 250, 247, 0.97) !important;
            border: 1px solid rgba(17, 17, 17, 0.4) !important;
        }

        .leaflet-popup-content {
            margin: 14px !important;
        }

        .up-popup {
            width: 285px;
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
            font-size: 11px;
            line-height: 1.8;
            font-weight: 700;
        }

        .up-popup-ratios span {
            color: #555;
            text-transform: uppercase;
        }

        .up-popup-ratios b {
            font-weight: 900;
        }
    </style>
    """

    map_obj.get_root().header.add_child(folium.Element(css))


def main() -> None:
    args = parse_args()

    input_csv = resolve_path(args.input_csv)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Reliability CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_columns = {
        "station_id",
        "station_name",
        "capacity",
        "lat",
        "lon",
        "observed_minutes",
        "available_minutes",
        "empty_minutes",
        "full_minutes",
        "offline_minutes",
        "rental_failure_ratio",
        "return_failure_ratio",
        "any_failure_ratio",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["lat", "lon"])
    df = df[df["observed_minutes"] >= args.min_observed_min].copy()

    if args.top_n > 0:
        df = df.sort_values("any_failure_ratio", ascending=False).head(args.top_n).copy()

    if df.empty:
        raise ValueError("No stations left after filtering.")

    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())
    median_observed = float(df["observed_minutes"].median())

    map_obj = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )

    add_layer(map_obj, df, "overall", show=True)
    add_layer(map_obj, df, "rent", show=False)
    add_layer(map_obj, df, "dock", show=False)

    folium.LayerControl(collapsed=False, position="topright").add_to(map_obj)

    add_css(map_obj)
    add_title(map_obj, args.window_label, len(df), median_observed)
    add_legend(map_obj)

    map_obj.save(out_path)

    print("Citi Bike reliability map built.")
    print(f"input_csv: {input_csv}")
    print(f"stations:  {len(df)}")
    print(f"out:       {out_path}")


if __name__ == "__main__":
    main()
