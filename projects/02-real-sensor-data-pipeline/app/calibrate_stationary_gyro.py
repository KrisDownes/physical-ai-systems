import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


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
    return df.pivot_table(
        index="timestamp_sec",
        columns="sensor_name",
        values="value",
        aggfunc="mean",
    ).reset_index()


def find_column(columns: list[str], target: str) -> str:
    target_lower = target.lower()

    matches = [
        col for col in columns
        if target_lower in col.lower()
    ]

    if not matches:
        raise ValueError(f"Could not find column matching: {target}")

    return matches[0]


def calibrate_gyro(wide: pd.DataFrame) -> dict:
    columns = list(wide.columns)

    gx_col = find_column(columns, "Gyroscope x")
    gy_col = find_column(columns, "Gyroscope y")
    gz_col = find_column(columns, "Gyroscope z")

    gx = wide[gx_col].to_numpy()
    gy = wide[gy_col].to_numpy()
    gz = wide[gz_col].to_numpy()

    gyro = np.column_stack([gx, gy, gz])

    mean_vec = gyro.mean(axis=0)
    std_vec = gyro.std(axis=0)

    magnitudes = np.linalg.norm(gyro, axis=1)

    mean_magnitude = float(magnitudes.mean())
    std_magnitude = float(magnitudes.std())

    # Convert rad/s to deg/s for human readability.
    mean_vec_deg = np.degrees(mean_vec)
    std_vec_deg = np.degrees(std_vec)

    # Drift estimate: if this bias were integrated for a period of time.
    drift_60_sec_rad = mean_vec * 60.0
    drift_10_min_rad = mean_vec * 600.0

    return {
        "gyro_columns": {
            "x": gx_col,
            "y": gy_col,
            "z": gz_col,
        },
        "mean_gyro_bias_rad_per_sec": {
            "x": float(mean_vec[0]),
            "y": float(mean_vec[1]),
            "z": float(mean_vec[2]),
        },
        "std_gyro_noise_rad_per_sec": {
            "x": float(std_vec[0]),
            "y": float(std_vec[1]),
            "z": float(std_vec[2]),
        },
        "mean_gyro_bias_deg_per_sec": {
            "x": float(mean_vec_deg[0]),
            "y": float(mean_vec_deg[1]),
            "z": float(mean_vec_deg[2]),
        },
        "std_gyro_noise_deg_per_sec": {
            "x": float(std_vec_deg[0]),
            "y": float(std_vec_deg[1]),
            "z": float(std_vec_deg[2]),
        },
        "mean_angular_speed_rad_per_sec": mean_magnitude,
        "std_angular_speed_rad_per_sec": std_magnitude,
        "estimated_angle_drift_after_60_sec_deg": {
            "x": float(np.degrees(drift_60_sec_rad[0])),
            "y": float(np.degrees(drift_60_sec_rad[1])),
            "z": float(np.degrees(drift_60_sec_rad[2])),
        },
        "estimated_angle_drift_after_10_min_deg": {
            "x": float(np.degrees(drift_10_min_rad[0])),
            "y": float(np.degrees(drift_10_min_rad[1])),
            "z": float(np.degrees(drift_10_min_rad[2])),
        },
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_measurements(Path(args.db), args.run_id)
    wide = pivot_sensor_data(df)
    result = calibrate_gyro(wide)

    output_path = output_dir / f"{args.run_id}_stationary_gyro_calibration.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved calibration to {output_path}")


if __name__ == "__main__":
    main()