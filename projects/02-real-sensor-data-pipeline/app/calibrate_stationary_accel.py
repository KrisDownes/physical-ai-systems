import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


STANDARD_GRAVITY = 9.80665


def load_measurements(db_path: Path, run_id: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        """
        SELECT timestamp_sec, sensor_name, value
        FROM measurements
        WHERE run_id = ?
        ORDER BY timestamp_sec ASC
        """,
        conn,
        params=(run_id,),
    )

    conn.close()

    if df.empty:
        raise ValueError(f"No data found for run_id={run_id}")

    return df


def pivot_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(
        index="timestamp_sec",
        columns="sensor_name",
        values="value",
        aggfunc="mean",
    ).reset_index()

    return wide


def find_column(columns: list[str], target: str) -> str:
    target_lower = target.lower()

    matches = [
        col for col in columns
        if target_lower in col.lower()
    ]

    if not matches:
        raise ValueError(f"Could not find column matching: {target}")

    return matches[0]


def calibrate_accel(wide: pd.DataFrame) -> dict:
    columns = list(wide.columns)

    ax_col = find_column(columns, "Acceleration x")
    ay_col = find_column(columns, "Acceleration y")
    az_col = find_column(columns, "Acceleration z")

    ax = wide[ax_col].to_numpy()
    ay = wide[ay_col].to_numpy()
    az = wide[az_col].to_numpy()

    accel = np.column_stack([ax, ay, az])

    mean_vec = accel.mean(axis=0)
    std_vec = accel.std(axis=0)

    magnitudes = np.linalg.norm(accel, axis=1)

    mean_magnitude = float(magnitudes.mean())
    std_magnitude = float(magnitudes.std())

    mean_vector_magnitude = float(np.linalg.norm(mean_vec))

    gravity_error = mean_magnitude - STANDARD_GRAVITY

    # Approximate tilt angles.
    # These are small-angle-ish interpretations:
    # roll around x and pitch around y based on gravity vector.
    ax_mean, ay_mean, az_mean = mean_vec

    roll_rad = np.arctan2(ay_mean, az_mean)
    pitch_rad = np.arctan2(-ax_mean, np.sqrt(ay_mean ** 2 + az_mean ** 2))

    roll_deg = float(np.degrees(roll_rad))
    pitch_deg = float(np.degrees(pitch_rad))

    return {
        "accel_columns": {
            "x": ax_col,
            "y": ay_col,
            "z": az_col,
        },
        "standard_gravity_mps2": STANDARD_GRAVITY,
        "mean_accel_vector_mps2": {
            "x": float(mean_vec[0]),
            "y": float(mean_vec[1]),
            "z": float(mean_vec[2]),
        },
        "std_accel_vector_mps2": {
            "x": float(std_vec[0]),
            "y": float(std_vec[1]),
            "z": float(std_vec[2]),
        },
        "mean_accel_magnitude_mps2": mean_magnitude,
        "std_accel_magnitude_mps2": std_magnitude,
        "mean_vector_magnitude_mps2": mean_vector_magnitude,
        "gravity_error_mps2": float(gravity_error),
        "estimated_roll_deg": roll_deg,
        "estimated_pitch_deg": pitch_deg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="/mnt/d/PhoneTelemetry/analysis",
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_measurements(db_path, args.run_id)
    wide = pivot_sensor_data(df)
    result = calibrate_accel(wide)

    output_path = output_dir / f"{args.run_id}_stationary_accel_calibration.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved calibration to {output_path}")


if __name__ == "__main__":
    main()