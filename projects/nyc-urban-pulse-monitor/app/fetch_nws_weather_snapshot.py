from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NWS_BASE_URL = "https://api.weather.gov"

DEFAULT_LOCATIONS = {
    "manhattan_midtown": (40.7580, -73.9855),
    "brooklyn_downtown": (40.6955, -73.9925),
    "queens_jackson_heights": (40.7557, -73.8831),
    "bronx_grand_concourse": (40.8262, -73.9227),
    "staten_island_st_george": (40.6437, -74.0736),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NWS weather snapshots for NYC monitoring points."
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "nws" / "weather",
    )

    parser.add_argument(
        "--locations",
        nargs="+",
        default=list(DEFAULT_LOCATIONS.keys()),
        choices=list(DEFAULT_LOCATIONS.keys()),
        help="Default NYC monitoring locations to fetch.",
    )

    parser.add_argument(
        "--point",
        action="append",
        default=[],
        help="Optional custom point as name,lat,lon. Can be repeated.",
    )

    parser.add_argument(
        "--max-stations",
        type=int,
        default=2,
        help="How many nearby observation stations to try per point.",
    )

    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
    )

    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_slug(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


def fetch_json(url: str, timeout_sec: float) -> dict[str, Any]:
    response = requests.get(
        url,
        timeout=timeout_sec,
        headers={
            "Accept": "application/geo+json, application/json",
            "User-Agent": "nyc-urban-pulse-monitor/0.1 (learning project)",
        },
    )
    response.raise_for_status()
    return response.json()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def parse_custom_points(values: list[str]) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}

    for value in values:
        try:
            name, lat_str, lon_str = value.split(",", 2)
            points[name.strip()] = (float(lat_str), float(lon_str))
        except Exception as exc:
            raise ValueError(
                f"Invalid --point value {value!r}. Expected name,lat,lon"
            ) from exc

    return points


def fetch_location(
    *,
    location_id: str,
    lat: float,
    lon: float,
    timeout_sec: float,
    max_stations: int,
) -> dict[str, Any]:
    point_url = f"{NWS_BASE_URL}/points/{lat:.4f},{lon:.4f}"
    point_doc = fetch_json(point_url, timeout_sec=timeout_sec)

    point_props = point_doc.get("properties", {})
    forecast_hourly_url = point_props.get("forecastHourly")
    forecast_url = point_props.get("forecast")
    observation_stations_url = point_props.get("observationStations")

    hourly_forecast = None
    daily_forecast = None
    observation_stations = None
    latest_observations = []
    active_alerts = None

    if forecast_hourly_url:
        hourly_forecast = fetch_json(forecast_hourly_url, timeout_sec=timeout_sec)

    if forecast_url:
        daily_forecast = fetch_json(forecast_url, timeout_sec=timeout_sec)

    if observation_stations_url:
        observation_stations = fetch_json(
            observation_stations_url,
            timeout_sec=timeout_sec,
        )

        station_features = observation_stations.get("features", [])[:max_stations]

        for station_feature in station_features:
            station_id_url = station_feature.get("id")
            station_props = station_feature.get("properties", {})

            if not station_id_url:
                continue

            latest_url = f"{station_id_url}/observations/latest"

            try:
                observation_doc = fetch_json(latest_url, timeout_sec=timeout_sec)
                latest_observations.append(
                    {
                        "station": station_props,
                        "station_url": station_id_url,
                        "latest_observation_url": latest_url,
                        "observation": observation_doc,
                    }
                )
            except Exception as exc:
                latest_observations.append(
                    {
                        "station": station_props,
                        "station_url": station_id_url,
                        "latest_observation_url": latest_url,
                        "error": str(exc),
                    }
                )

    alerts_url = f"{NWS_BASE_URL}/alerts/active?point={lat:.4f},{lon:.4f}"
    active_alerts = fetch_json(alerts_url, timeout_sec=timeout_sec)

    return {
        "location_id": location_id,
        "lat": lat,
        "lon": lon,
        "point_url": point_url,
        "point": point_doc,
        "hourly_forecast": hourly_forecast,
        "daily_forecast": daily_forecast,
        "observation_stations": observation_stations,
        "latest_observations": latest_observations,
        "active_alerts": active_alerts,
    }


def main() -> None:
    args = parse_args()

    locations: dict[str, tuple[float, float]] = {
        name: DEFAULT_LOCATIONS[name] for name in args.locations
    }
    locations.update(parse_custom_points(args.point))

    observed_at_utc = utc_now()
    slug = timestamp_slug(observed_at_utc)

    fetched_locations = []

    for location_id, (lat, lon) in locations.items():
        print(f"Fetching NWS weather for {location_id}: {lat}, {lon}")

        fetched_locations.append(
            fetch_location(
                location_id=location_id,
                lat=lat,
                lon=lon,
                timeout_sec=args.timeout_sec,
                max_stations=args.max_stations,
            )
        )

    output_path = args.out_dir / f"nws_weather_{slug}.json"

    wrapped_payload = {
        "source_name": "nws_weather",
        "source_url": NWS_BASE_URL,
        "observed_at_utc": observed_at_utc.isoformat(),
        "fetched_at_utc": utc_now().isoformat(),
        "locations": fetched_locations,
    }

    write_json(output_path, wrapped_payload)

    print("\nNWS weather snapshot saved.")
    print(f"saved:     {output_path}")
    print(f"locations: {len(fetched_locations)}")

    for loc in fetched_locations:
        hourly_periods = (
            loc.get("hourly_forecast", {})
            .get("properties", {})
            .get("periods", [])
        )
        alert_count = len(loc.get("active_alerts", {}).get("features", []))
        obs_count = len(loc.get("latest_observations", []))

        print(
            f"- {loc['location_id']}: "
            f"hourly_periods={len(hourly_periods)} "
            f"latest_observations={obs_count} "
            f"active_alerts={alert_count}"
        )


if __name__ == "__main__":
    main()
