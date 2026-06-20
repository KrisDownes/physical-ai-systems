from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest raw NWS weather snapshots into DuckDB."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "nws" / "weather",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "urban_pulse.duckdb",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None

    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    return dt


def quantity_value(obj: Any) -> float | None:
    if not isinstance(obj, dict):
        return None

    value = obj.get("value")

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def c_to_f(value_c: float | None) -> float | None:
    if value_c is None:
        return None
    return value_c * 9.0 / 5.0 + 32.0


def mps_to_mph(value_mps: float | None) -> float | None:
    if value_mps is None:
        return None
    return value_mps * 2.2369362921


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None

    try:
        return int(float(value))
    except Exception:
        return None


def parse_wind_speed_mph(value: str | None) -> float | None:
    if not value:
        return None

    nums = re.findall(r"\d+(?:\.\d+)?", str(value))

    if not nums:
        return None

    # For strings like "5 to 10 mph", use the high end.
    return max(float(x) for x in nums)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True)


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_nws_weather_snapshots (
            snapshot_id VARCHAR,
            source_name VARCHAR,
            source_url VARCHAR,
            observed_at_utc TIMESTAMP,
            fetched_at_utc TIMESTAMP,
            raw_file_path VARCHAR,
            location_count INTEGER,
            ingested_at_utc TIMESTAMP
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_weather_locations (
            location_id VARCHAR,
            location_name VARCHAR,
            lat DOUBLE,
            lon DOUBLE,
            nws_office VARCHAR,
            grid_id VARCHAR,
            grid_x INTEGER,
            grid_y INTEGER,
            forecast_zone VARCHAR,
            county_zone VARCHAR,
            fire_weather_zone VARCHAR,
            forecast_hourly_url VARCHAR,
            observation_stations_url VARCHAR,
            last_seen_snapshot_id VARCHAR,
            last_seen_at_utc TIMESTAMP,
            raw_file_path VARCHAR
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_nws_hourly_forecast (
            snapshot_id VARCHAR,
            location_id VARCHAR,
            observed_at_utc TIMESTAMP,
            forecast_generated_at_utc TIMESTAMP,
            forecast_update_time_utc TIMESTAMP,
            period_number INTEGER,
            period_name VARCHAR,
            forecast_start_utc TIMESTAMP,
            forecast_end_utc TIMESTAMP,
            is_daytime BOOLEAN,
            temperature_f DOUBLE,
            precipitation_probability_pct DOUBLE,
            relative_humidity_pct DOUBLE,
            dewpoint_c DOUBLE,
            wind_speed_mph_est DOUBLE,
            wind_direction VARCHAR,
            short_forecast VARCHAR,
            detailed_forecast VARCHAR,
            raw_period_json VARCHAR
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_nws_latest_observations (
            snapshot_id VARCHAR,
            location_id VARCHAR,
            observed_at_utc TIMESTAMP,
            station_id VARCHAR,
            station_name VARCHAR,
            station_url VARCHAR,
            observation_time_utc TIMESTAMP,
            temperature_c DOUBLE,
            temperature_f DOUBLE,
            dewpoint_c DOUBLE,
            relative_humidity_pct DOUBLE,
            wind_speed_mps DOUBLE,
            wind_speed_mph DOUBLE,
            wind_gust_mps DOUBLE,
            wind_gust_mph DOUBLE,
            barometric_pressure_pa DOUBLE,
            text_description VARCHAR,
            raw_observation_json VARCHAR
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_nws_active_alerts (
            snapshot_id VARCHAR,
            location_id VARCHAR,
            observed_at_utc TIMESTAMP,
            alert_id VARCHAR,
            sent_utc TIMESTAMP,
            effective_utc TIMESTAMP,
            onset_utc TIMESTAMP,
            expires_utc TIMESTAMP,
            ends_utc TIMESTAMP,
            status VARCHAR,
            message_type VARCHAR,
            category VARCHAR,
            severity VARCHAR,
            certainty VARCHAR,
            urgency VARCHAR,
            event VARCHAR,
            headline VARCHAR,
            area_desc VARCHAR,
            description VARCHAR,
            instruction VARCHAR,
            raw_alert_json VARCHAR
        );
        """
    )


def upsert_raw_snapshot(
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id: str,
    doc: dict[str, Any],
    path: Path,
) -> None:
    row = {
        "snapshot_id": snapshot_id,
        "source_name": doc.get("source_name"),
        "source_url": doc.get("source_url"),
        "observed_at_utc": parse_iso_timestamp(doc.get("observed_at_utc")),
        "fetched_at_utc": parse_iso_timestamp(doc.get("fetched_at_utc")),
        "raw_file_path": str(path),
        "location_count": len(doc.get("locations", [])),
        "ingested_at_utc": datetime.now(timezone.utc).replace(tzinfo=None),
    }

    df = pd.DataFrame([row])
    con.register("tmp_raw_nws_weather", df)

    con.execute(
        "DELETE FROM raw_nws_weather_snapshots WHERE snapshot_id = ?",
        [snapshot_id],
    )

    con.execute(
        """
        INSERT INTO raw_nws_weather_snapshots
        SELECT * FROM tmp_raw_nws_weather;
        """
    )


def upsert_dim_locations(
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id: str,
    doc: dict[str, Any],
    path: Path,
) -> int:
    observed_at_utc = parse_iso_timestamp(doc.get("observed_at_utc"))
    rows = []

    for loc in doc.get("locations", []):
        point_props = loc.get("point", {}).get("properties", {})

        rows.append(
            {
                "location_id": loc.get("location_id"),
                "location_name": loc.get("location_id"),
                "lat": safe_float(loc.get("lat")),
                "lon": safe_float(loc.get("lon")),
                "nws_office": point_props.get("cwa"),
                "grid_id": point_props.get("gridId"),
                "grid_x": safe_int(point_props.get("gridX")),
                "grid_y": safe_int(point_props.get("gridY")),
                "forecast_zone": point_props.get("forecastZone"),
                "county_zone": point_props.get("county"),
                "fire_weather_zone": point_props.get("fireWeatherZone"),
                "forecast_hourly_url": point_props.get("forecastHourly"),
                "observation_stations_url": point_props.get("observationStations"),
                "last_seen_snapshot_id": snapshot_id,
                "last_seen_at_utc": observed_at_utc,
                "raw_file_path": str(path),
            }
        )

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    con.register("tmp_weather_locations", df)

    con.execute(
        """
        DELETE FROM dim_weather_locations
        WHERE location_id IN (
            SELECT location_id FROM tmp_weather_locations
        );
        """
    )

    con.execute(
        """
        INSERT INTO dim_weather_locations
        SELECT * FROM tmp_weather_locations;
        """
    )

    return len(rows)


def insert_hourly_forecasts(
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id: str,
    doc: dict[str, Any],
) -> int:
    observed_at_utc = parse_iso_timestamp(doc.get("observed_at_utc"))
    rows = []

    for loc in doc.get("locations", []):
        location_id = loc.get("location_id")
        forecast = loc.get("hourly_forecast") or {}
        forecast_props = forecast.get("properties", {})

        generated_at = parse_iso_timestamp(forecast_props.get("generatedAt"))
        update_time = parse_iso_timestamp(forecast_props.get("updateTime"))

        for period in forecast_props.get("periods", []):
            temp = safe_float(period.get("temperature"))
            temp_unit = period.get("temperatureUnit")

            if temp is not None and temp_unit == "C":
                temp_f = c_to_f(temp)
            else:
                temp_f = temp

            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "location_id": location_id,
                    "observed_at_utc": observed_at_utc,
                    "forecast_generated_at_utc": generated_at,
                    "forecast_update_time_utc": update_time,
                    "period_number": safe_int(period.get("number")),
                    "period_name": period.get("name"),
                    "forecast_start_utc": parse_iso_timestamp(period.get("startTime")),
                    "forecast_end_utc": parse_iso_timestamp(period.get("endTime")),
                    "is_daytime": period.get("isDaytime"),
                    "temperature_f": temp_f,
                    "precipitation_probability_pct": quantity_value(
                        period.get("probabilityOfPrecipitation")
                    ),
                    "relative_humidity_pct": quantity_value(
                        period.get("relativeHumidity")
                    ),
                    "dewpoint_c": quantity_value(period.get("dewpoint")),
                    "wind_speed_mph_est": parse_wind_speed_mph(
                        period.get("windSpeed")
                    ),
                    "wind_direction": period.get("windDirection"),
                    "short_forecast": period.get("shortForecast"),
                    "detailed_forecast": period.get("detailedForecast"),
                    "raw_period_json": json_dumps(period),
                }
            )

    con.execute(
        "DELETE FROM fact_nws_hourly_forecast WHERE snapshot_id = ?",
        [snapshot_id],
    )

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    con.register("tmp_nws_hourly_forecast", df)

    con.execute(
        """
        INSERT INTO fact_nws_hourly_forecast
        SELECT * FROM tmp_nws_hourly_forecast;
        """
    )

    return len(rows)


def insert_latest_observations(
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id: str,
    doc: dict[str, Any],
) -> int:
    observed_at_utc = parse_iso_timestamp(doc.get("observed_at_utc"))
    rows = []

    for loc in doc.get("locations", []):
        location_id = loc.get("location_id")

        for obs_entry in loc.get("latest_observations", []):
            if "observation" not in obs_entry:
                continue

            station = obs_entry.get("station", {})
            obs_doc = obs_entry.get("observation", {})
            props = obs_doc.get("properties", {})

            station_id = station.get("stationIdentifier")
            station_name = station.get("name")
            station_url = obs_entry.get("station_url")

            temp_c = quantity_value(props.get("temperature"))
            dewpoint_c = quantity_value(props.get("dewpoint"))
            wind_speed_mps = quantity_value(props.get("windSpeed"))
            wind_gust_mps = quantity_value(props.get("windGust"))

            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "location_id": location_id,
                    "observed_at_utc": observed_at_utc,
                    "station_id": station_id,
                    "station_name": station_name,
                    "station_url": station_url,
                    "observation_time_utc": parse_iso_timestamp(
                        props.get("timestamp")
                    ),
                    "temperature_c": temp_c,
                    "temperature_f": c_to_f(temp_c),
                    "dewpoint_c": dewpoint_c,
                    "relative_humidity_pct": quantity_value(
                        props.get("relativeHumidity")
                    ),
                    "wind_speed_mps": wind_speed_mps,
                    "wind_speed_mph": mps_to_mph(wind_speed_mps),
                    "wind_gust_mps": wind_gust_mps,
                    "wind_gust_mph": mps_to_mph(wind_gust_mps),
                    "barometric_pressure_pa": quantity_value(
                        props.get("barometricPressure")
                    ),
                    "text_description": props.get("textDescription"),
                    "raw_observation_json": json_dumps(obs_doc),
                }
            )

    con.execute(
        "DELETE FROM fact_nws_latest_observations WHERE snapshot_id = ?",
        [snapshot_id],
    )

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    con.register("tmp_nws_latest_observations", df)

    con.execute(
        """
        INSERT INTO fact_nws_latest_observations
        SELECT * FROM tmp_nws_latest_observations;
        """
    )

    return len(rows)


def insert_active_alerts(
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id: str,
    doc: dict[str, Any],
) -> int:
    observed_at_utc = parse_iso_timestamp(doc.get("observed_at_utc"))
    rows = []

    for loc in doc.get("locations", []):
        location_id = loc.get("location_id")
        alerts = loc.get("active_alerts") or {}

        for feature in alerts.get("features", []):
            props = feature.get("properties", {})

            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "location_id": location_id,
                    "observed_at_utc": observed_at_utc,
                    "alert_id": feature.get("id") or props.get("id"),
                    "sent_utc": parse_iso_timestamp(props.get("sent")),
                    "effective_utc": parse_iso_timestamp(props.get("effective")),
                    "onset_utc": parse_iso_timestamp(props.get("onset")),
                    "expires_utc": parse_iso_timestamp(props.get("expires")),
                    "ends_utc": parse_iso_timestamp(props.get("ends")),
                    "status": props.get("status"),
                    "message_type": props.get("messageType"),
                    "category": props.get("category"),
                    "severity": props.get("severity"),
                    "certainty": props.get("certainty"),
                    "urgency": props.get("urgency"),
                    "event": props.get("event"),
                    "headline": props.get("headline"),
                    "area_desc": props.get("areaDesc"),
                    "description": props.get("description"),
                    "instruction": props.get("instruction"),
                    "raw_alert_json": json_dumps(feature),
                }
            )

    con.execute(
        "DELETE FROM fact_nws_active_alerts WHERE snapshot_id = ?",
        [snapshot_id],
    )

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    con.register("tmp_nws_active_alerts", df)

    con.execute(
        """
        INSERT INTO fact_nws_active_alerts
        SELECT * FROM tmp_nws_active_alerts;
        """
    )

    return len(rows)


def main() -> None:
    args = parse_args()

    raw_dir = resolve_path(args.raw_dir)
    db_path = resolve_path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("nws_weather_*.json"))

    if not files:
        raise FileNotFoundError(f"No NWS weather snapshots found in: {raw_dir}")

    con = duckdb.connect(str(db_path))
    create_tables(con)

    print("NWS weather ingest")
    print(f"raw_dir: {raw_dir}")
    print(f"db:      {db_path}")
    print(f"files:   {len(files)}")

    total_locations = 0
    total_forecasts = 0
    total_observations = 0
    total_alerts = 0

    for path in files:
        doc = read_json(path)
        snapshot_id = path.stem

        upsert_raw_snapshot(con, snapshot_id=snapshot_id, doc=doc, path=path)

        location_rows = upsert_dim_locations(
            con,
            snapshot_id=snapshot_id,
            doc=doc,
            path=path,
        )
        forecast_rows = insert_hourly_forecasts(
            con,
            snapshot_id=snapshot_id,
            doc=doc,
        )
        observation_rows = insert_latest_observations(
            con,
            snapshot_id=snapshot_id,
            doc=doc,
        )
        alert_rows = insert_active_alerts(
            con,
            snapshot_id=snapshot_id,
            doc=doc,
        )

        total_locations += location_rows
        total_forecasts += forecast_rows
        total_observations += observation_rows
        total_alerts += alert_rows

        print(
            f"{path.name}: "
            f"locations={location_rows} "
            f"hourly_forecasts={forecast_rows} "
            f"observations={observation_rows} "
            f"alerts={alert_rows}"
        )

    print("\nIngest complete.")
    print(f"location rows processed:    {total_locations}")
    print(f"hourly forecasts processed: {total_forecasts}")
    print(f"observations processed:     {total_observations}")
    print(f"alerts processed:           {total_alerts}")

    print("\nWarehouse counts:")
    print(
        con.execute(
            """
            SELECT COUNT(*) AS weather_locations
            FROM dim_weather_locations
            """
        ).fetchdf()
    )

    print(
        con.execute(
            """
            SELECT
                COUNT(*) AS hourly_forecast_rows,
                COUNT(DISTINCT snapshot_id) AS snapshots,
                COUNT(DISTINCT location_id) AS locations,
                MIN(forecast_start_utc) AS first_forecast_start,
                MAX(forecast_start_utc) AS last_forecast_start
            FROM fact_nws_hourly_forecast
            """
        ).fetchdf()
    )

    print("\nLatest forecast sample:")
    print(
        con.execute(
            """
            SELECT
                location_id,
                observed_at_utc,
                forecast_start_utc,
                temperature_f,
                precipitation_probability_pct,
                wind_speed_mph_est,
                short_forecast
            FROM fact_nws_hourly_forecast
            ORDER BY observed_at_utc DESC, location_id, forecast_start_utc
            LIMIT 10
            """
        ).fetchdf()
    )


if __name__ == "__main__":
    main()
