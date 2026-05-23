import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    matches = [col for col in columns if target_lower in col.lower()]
    if not matches:
        raise ValueError(f"Could not find column matching: {target}")
    return matches[0]

def integrate_gyro_axis(wide: pd.DataFrame, axis: str, bias_rad_s: float, expected_angle_deg: str | None) -> dict:
    columns = list(wide.columns)
    gyro_col = find_column(columns, f"Gyroscope {axis}")
    time_col = "timestamp_sec"

    time = wide[time_col].to_numpy()
    gyro_raw = wide[gyro_col].to_numpy()

    time = time - time[0]
    dt = np.diff(time, prepend=time[0])
    gyro_calibrated = gyro_raw - bias_rad_s

    # Angular velocity integration using cumulative trapezoidal rule
    angle_rad = np.cumsum(gyro_calibrated * dt)
    angle_deg = np.degrees(angle_rad)

    final_angle_rad = angle_rad[-1]
    final_angle_deg = angle_deg[-1]

    

    if expected_angle_deg is not None:
        expected_angle_deg = float(expected_angle_deg)
        angle_error_deg = final_angle_deg - expected_angle_deg
        angle_abs_error_deg = abs(angle_error_deg)
        if expected_angle_deg != 0:
            angle_percent_error = 100.0 * angle_abs_error_deg / abs(expected_angle_deg)

    duration_sec = time[-1] - time[0]
    mean_dt = float(np.mean(dt[1:])) if len(dt) > 1 else 0.0
    sample_rate_hz = 1.0 / mean_dt if mean_dt > 0 else 0.0

    return {
        "time": time,
        "gyro_raw": gyro_raw,
        "gyro_corrected": gyro_calibrated,
        "dt": dt,
        "angle_rad": angle_rad,
        "angle_deg": angle_deg,
        "summary": {
            "gyro_column": gyro_col,
            "axis": axis,
            "bias_rad_per_sec": bias_rad_s,
            "duration_sec": duration_sec,
            "sample_count": int(len(time)),
            "mean_dt_sec": mean_dt,
            "sample_rate_hz": sample_rate_hz,
            "final_angle_rad": final_angle_rad,
            "final_angle_deg": final_angle_deg,
            "expected_angle_deg": expected_angle_deg,
            "angle_error_deg": angle_error_deg,
            "angle_abs_error_deg": angle_abs_error_deg,
            "angle_percent_error": angle_percent_error,
        },
    }

def save_outputs(
    result: dict,
    run_id: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    time = result["time"]
    gyro_raw = result["gyro_raw"]
    gyro_corrected = result["gyro_corrected"]
    angle_deg = result["angle_deg"]
    summary = result["summary"]
    axis = summary["axis"]

    summary_path = output_dir / f"{run_id}_gyro_angle_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    plt.figure(figsize=(12, 5))
    plt.plot(time, gyro_raw, label="raw gyro measurement")
    plt.plot(time, gyro_corrected, label="bias-corrected gyro")
    plt.xlabel("Time (sec)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.title(f"{run_id}: Gyro {axis.upper()} Angular Velocity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{run_id}_gyro_{axis}.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(time, angle_deg, label="integrated angle")
    plt.xlabel("Time (sec)")
    plt.ylabel("Angle (degrees)")
    plt.title(f"{run_id}: Integrated {axis.upper()} Rotation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{run_id}_angle_{axis}.png")
    plt.close()

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default="/mnt/d/PhoneTelemetry/db/telemetry.db", help="Path to SQLite database")
    parser.add_argument("--run-id", required=True, help="Run ID to process")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z", help="Axis to integrate (default: z)")
    parser.add_argument("--bias", type=float, default=None, help="Gyro bias in rad/s")
    parser.add_argument("--expected-angle-deg", type=float, default=None, help="Expected final angle in degrees")
    parser.add_argument("--output-dir", type=Path, default="/mnt/d/PhoneTelemetry/analysis", help="Directory to save outputs")
    args = parser.parse_args()
    df = load_measurements(Path(args.db), args.run_id)
    wide = pivot_sensor_data(df)
    result = integrate_gyro_axis(wide, axis= args.axis, bias_rad_s=args.bias, expected_angle_deg=args.expected_angle_deg)
    save_outputs(result, args.run_id, args.output_dir)
    dt = np.diff(wide["timestamp_sec"], prepend=wide["timestamp_sec"].iloc[0])
    
    
if __name__ == "__main__":
    main()

