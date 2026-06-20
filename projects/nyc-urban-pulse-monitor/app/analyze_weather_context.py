from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SEVERITY_SCORE = {
    "Unknown": 0.0,
    "Minor": 10.0,
    "Moderate": 25.0,
    "Severe": 45.0,
    "Extreme": 65.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build current weather context and stress scores."
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

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def score_precip(precip_pct: float | None) -> float:
    if pd.isna(precip_pct):
        return 0.0
    return clamp(float(precip_pct) * 0.40, 0.0, 40.0)


def score_wind(wind_mph: float | None) -> float:
    if pd.isna(wind_mph):
        return 0.0

    wind_mph = float(wind_mph)

    if wind_mph <= 15.0:
        return 0.0

    return clamp((wind_mph - 15.0) * 1.5, 0.0, 30.0)


def score_temperature(temp_f: float | None) -> float:
    if pd.isna(temp_f):
        return 0.0

    temp_f = float(temp_f)

    if temp_f >= 85.0:
        return clamp((temp_f - 85.0) * 2.0, 0.0, 25.0)

    if temp_f <= 32.0:
        return clamp((32.0 - temp_f) * 1.5, 0.0, 25.0)

    return 0.0


def classify_weather(row: pd.Series) -> str:
    alerts = int(row.get("active_alert_count", 0) or 0)
    precip = float(row.get("precipitation_probability_pct") or 0.0)
    wind = float(row.get("wind_speed_mph_est") or 0.0)
    temp = row.get("temperature_f")
    score = float(row.get("weather_stress_score") or 0.0)

    if alerts > 0 and score >= 50:
        return "active alert / high weather stress"

    if alerts > 0:
        return "active weather alert"

    if precip >= 50 and wind >= 20:
        return "wet and windy"

    if precip >= 50:
        return "rain risk"

    if wind >= 25:
        return "wind stress"

    if pd.notna(temp) and float(temp) >= 90:
        return "heat stress"

    if pd.notna(temp) and float(temp) <= 32:
        return "cold stress"

    if score >= 35:
        return "moderate weather stress"

    return "low weather stress"


def main() -> None:
    args = parse_args()

    db_path = resolve_path(args.db)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")

    con = duckdb.connect(str(db_path))

    current_forecast = con.execute(
        """
        WITH latest_location_snapshot AS (
            SELECT
                location_id,
                MAX(observed_at_utc) AS latest_observed_at_utc
            FROM fact_nws_hourly_forecast
            GROUP BY location_id
        ),

        latest_forecast AS (
            SELECT f.*
            FROM fact_nws_hourly_forecast f
            JOIN latest_location_snapshot l
                ON f.location_id = l.location_id
                AND f.observed_at_utc = l.latest_observed_at_utc
        ),

        ranked_periods AS (
            SELECT
                *,
                CASE
                    WHEN
                        forecast_start_utc <= observed_at_utc
                        AND forecast_end_utc > observed_at_utc
                    THEN 0
                    WHEN forecast_start_utc > observed_at_utc
                    THEN 1
                    ELSE 2
                END AS period_priority,

                ABS(DATE_DIFF('minute', observed_at_utc, forecast_start_utc))
                    AS minutes_from_snapshot,

                ROW_NUMBER() OVER (
                    PARTITION BY location_id
                    ORDER BY
                        CASE
                            WHEN
                                forecast_start_utc <= observed_at_utc
                                AND forecast_end_utc > observed_at_utc
                            THEN 0
                            WHEN forecast_start_utc > observed_at_utc
                            THEN 1
                            ELSE 2
                        END,
                        ABS(DATE_DIFF('minute', observed_at_utc, forecast_start_utc))
                ) AS rn
            FROM latest_forecast
        )

        SELECT
            location_id,
            observed_at_utc,
            forecast_start_utc,
            forecast_end_utc,
            temperature_f,
            precipitation_probability_pct,
            relative_humidity_pct,
            dewpoint_c,
            wind_speed_mph_est,
            wind_direction,
            short_forecast
        FROM ranked_periods
        WHERE rn = 1
        ORDER BY location_id
        """
    ).fetchdf()

    latest_obs = con.execute(
        """
        WITH ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY location_id
                    ORDER BY observation_time_utc DESC NULLS LAST, observed_at_utc DESC
                ) AS rn
            FROM fact_nws_latest_observations
        )
        SELECT
            location_id,
            station_id,
            station_name,
            observation_time_utc,
            temperature_f AS observed_temperature_f,
            relative_humidity_pct AS observed_relative_humidity_pct,
            wind_speed_mph AS observed_wind_speed_mph,
            text_description AS observed_condition
        FROM ranked
        WHERE rn = 1
        """
    ).fetchdf()

    latest_snapshot = con.execute(
        """
        SELECT
            location_id,
            MAX(observed_at_utc) AS latest_observed_at_utc
        FROM fact_nws_hourly_forecast
        GROUP BY location_id
        """
    ).fetchdf()

    alerts = con.execute(
        """
        SELECT
            a.location_id,
            COUNT(*) AS active_alert_count,
            MAX(
                CASE severity
                    WHEN 'Extreme' THEN 65.0
                    WHEN 'Severe' THEN 45.0
                    WHEN 'Moderate' THEN 25.0
                    WHEN 'Minor' THEN 10.0
                    ELSE 0.0
                END
            ) AS alert_component,
            STRING_AGG(DISTINCT event, '; ') AS active_alert_events
        FROM fact_nws_active_alerts a
        JOIN (
            SELECT
                location_id,
                MAX(observed_at_utc) AS latest_observed_at_utc
            FROM fact_nws_hourly_forecast
            GROUP BY location_id
        ) l
            ON a.location_id = l.location_id
            AND a.observed_at_utc = l.latest_observed_at_utc
        GROUP BY a.location_id
        """
    ).fetchdf()

    context = current_forecast.merge(
        latest_obs,
        on="location_id",
        how="left",
    ).merge(
        alerts,
        on="location_id",
        how="left",
    ).merge(
        latest_snapshot,
        on="location_id",
        how="left",
    )

    if context.empty:
        raise ValueError("No current weather context rows were produced.")

    context["active_alert_count"] = context["active_alert_count"].fillna(0).astype(int)
    context["alert_component"] = context["alert_component"].fillna(0.0)
    context["active_alert_events"] = context["active_alert_events"].fillna("")

    precip_components = []
    wind_components = []
    temp_components = []
    total_scores = []

    for _, row in context.iterrows():
        precip_component = score_precip(row.get("precipitation_probability_pct"))
        wind_component = score_wind(row.get("wind_speed_mph_est"))
        temp_component = score_temperature(row.get("temperature_f"))
        alert_component = float(row.get("alert_component") or 0.0)

        total = clamp(
            precip_component
            + wind_component
            + temp_component
            + alert_component
        )

        precip_components.append(round(precip_component, 2))
        wind_components.append(round(wind_component, 2))
        temp_components.append(round(temp_component, 2))
        total_scores.append(round(total, 2))

    context["precip_component"] = precip_components
    context["wind_component"] = wind_components
    context["temperature_component"] = temp_components
    context["weather_stress_score"] = total_scores
    context["weather_stress_class"] = context.apply(classify_weather, axis=1)

    out_path = out_dir / "weather_current_context.csv"
    context.to_csv(out_path, index=False)

    city_summary = pd.DataFrame(
        [
            {
                "location_count": len(context),
                "avg_weather_stress_score": round(
                    context["weather_stress_score"].mean(),
                    2,
                ),
                "max_weather_stress_score": round(
                    context["weather_stress_score"].max(),
                    2,
                ),
                "max_precip_probability_pct": round(
                    context["precipitation_probability_pct"].fillna(0).max(),
                    2,
                ),
                "max_wind_speed_mph_est": round(
                    context["wind_speed_mph_est"].fillna(0).max(),
                    2,
                ),
                "active_alert_locations": int(
                    (context["active_alert_count"] > 0).sum()
                ),
                "dominant_weather_class": (
                    context["weather_stress_class"]
                    .value_counts()
                    .index[0]
                ),
            }
        ]
    )

    summary_path = out_dir / "weather_city_summary.csv"
    city_summary.to_csv(summary_path, index=False)

    print("Weather context analysis complete.")
    print(f"saved: {out_path}")
    print(f"saved: {summary_path}")

    print("\nCity summary:")
    print(city_summary)

    print("\nCurrent weather context:")
    print(
        context[
            [
                "location_id",
                "temperature_f",
                "precipitation_probability_pct",
                "wind_speed_mph_est",
                "short_forecast",
                "active_alert_count",
                "weather_stress_score",
                "weather_stress_class",
            ]
        ]
    )


if __name__ == "__main__":
    main()
