import pandas as pd
import sqlite3
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

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
        raise ValueError(f"Column not found for target: {target}")
    return matches[0]

def normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_col = "timestamp_sec"
    df[time_col] = df[time_col] - df[time_col].iloc[0]
    df = df.rename(columns={time_col: "time_sec"})
    return df

def rename_columns(df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    df = df.copy()
    columns = list(df.columns)
    specs = {
        "acc" : {
            "targets" : {
                "x": "acc_x",
                "y": "acc_y",
                "z": "acc_z",
            }
        },
        "gyro" : {
            "targets" : {
                "x": "gyro_x",
                "y": "gyro_y",
                "z": "gyro_z",
            }
        },
        "gravity" : {
            "targets" : {
                "Gravity x": "gravity_x",
                "Gravity y": "gravity_y",
                "Gravity z": "gravity_z",
            }
        }
    }
    if sensor_name not in specs:
        raise ValueError(f"Unknown sensor name for renaming: {sensor_name}")
    rename_map = {}
    for target, new_name in specs[sensor_name]["targets"].items():
        old_name = find_column(columns, target)
        rename_map[old_name] = new_name
    return df.rename(columns=rename_map)

def interpolate_time(source_df: pd.DataFrame, source_time_col: str, value_col: str, target_time: np.ndarray) -> np.ndarray:
    master_time = target_time
    source_time = source_df[source_time_col].to_numpy()
    source_values = source_df[value_col].to_numpy()
    interpolated_values = np.interp(master_time, source_time, source_values)
    return interpolated_values

def align_and_merge(wide_acc_df: pd.DataFrame, wide_gyro_df: pd.DataFrame, wide_gravity_df: pd.DataFrame) -> pd.DataFrame:
    time_col = "time_sec"
    end_time = min(wide_acc_df[time_col].iloc[-1], wide_gyro_df[time_col].iloc[-1], wide_gravity_df[time_col].iloc[-1])
    wide_acc_df = wide_acc_df[wide_acc_df[time_col] <= end_time].copy()
    wide_gyro_df = wide_gyro_df[wide_gyro_df[time_col] <= end_time].copy()
    wide_gravity_df = wide_gravity_df[wide_gravity_df[time_col] <= end_time].copy()
    master_time = wide_gyro_df[time_col].to_numpy()
    aligned = pd.DataFrame({
        "time_sec": master_time,
        "gyro_x": wide_gyro_df["gyro_x"].to_numpy(),
        "gyro_y": wide_gyro_df["gyro_y"].to_numpy(),
        "gyro_z": wide_gyro_df["gyro_z"].to_numpy(),
        "acc_x": interpolate_time(wide_acc_df, time_col, "acc_x", master_time),
        "acc_y": interpolate_time(wide_acc_df, time_col, "acc_y", master_time),
        "acc_z": interpolate_time(wide_acc_df, time_col, "acc_z", master_time),
        "gravity_x": interpolate_time(wide_gravity_df, time_col, "gravity_x", master_time),
        "gravity_y": interpolate_time(wide_gravity_df, time_col, "gravity_y", master_time),
        "gravity_z": interpolate_time(wide_gravity_df, time_col, "gravity_z", master_time),
    })
    return aligned
def summarize_aligned_data(aligned_df: pd.DataFrame) -> dict:
    time = aligned_df["time_sec"].to_numpy()
    dt = np.diff(time)

    acc_mag = np.sqrt(aligned_df["acc_x"]**2 + aligned_df["acc_y"]**2 + aligned_df["acc_z"]**2).to_numpy()
    gravity_mag = np.sqrt(aligned_df["gravity_x"]**2 + aligned_df["gravity_y"]**2 + aligned_df["gravity_z"]**2).to_numpy()
    gyro_mag = np.sqrt(aligned_df["gyro_x"]**2 + aligned_df["gyro_y"]**2 + aligned_df["gyro_z"]**2).to_numpy()

    return {
        "duration_sec": float(time[-1] - time[0]),
        "sample_count" : int(len(time)),
        "mean_dt_sec": float(np.mean(dt)),
        "sample_rate_hz": float(1.0/np.mean(dt)) if len(dt) > 0 else 0.0,
        "Mean accleration magnitude M/s^2": float(np.mean(acc_mag)),
        "Std accleration magnitude M/s^2": float(np.std(acc_mag)),
        "Mean gravity magnitude M/s^2": float(np.mean(gravity_mag)),
        "Std gravity magnitude M/s^2": float(np.std(gravity_mag)),
        "Mean gyro magnitude rad/s": float(np.mean(gyro_mag)),
        "Std gyro magnitude rad/s": float(np.std(gyro_mag)),
    }

def add_magnitude_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["gravity_mag"] = np.sqrt(df["gravity_x"]**2 + df["gravity_y"]**2 + df["gravity_z"]**2)
    df["gyro_mag"] = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2)
    return df

def detect_stationary_periods(df: pd.DataFrame, acc_threshold: float = 0.5, gyro_threshold: float = 0.1) -> pd.DataFrame:
    df = df.copy()
    window_sec = 1.0
    dt = np.diff(df["time_sec"]).mean() if len(df) > 1 else 0.01
    window_size = int(window_sec / dt)
    df["gyro_mag_mean"] = df["gyro_mag"].rolling(window=window_size, min_periods=window_size).mean()
    df["gyro_mag_std"] = df["gyro_mag"].rolling(window=window_size, min_periods=window_size).std()
    df["acc_mag_mean"] = df["acc_mag"].rolling(window=window_size, min_periods=window_size).mean()
    df["acc_mag_std"] = df["acc_mag"].rolling(window=window_size, min_periods=window_size).std()
    df["is_stationary"] = (df["gyro_mag_mean"] < gyro_threshold) & (df["gyro_mag_std"] < gyro_threshold) & (np.abs(df["acc_mag_mean"] - STANDARD_GRAVITY) < acc_threshold) & (df["acc_mag_std"] < acc_threshold)
    df["is_stationary"] = df["is_stationary"].fillna(False)
    return df
def calibrate_gyro_bias(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    stationary_gyro = df[df["is_stationary"]]
    if stationary_gyro.empty:
        raise ValueError("No stationary samples found; cannot estimate gyro bias")
    
    bias_x = stationary_gyro["gyro_x"].median()
    bias_y = stationary_gyro["gyro_y"].median()
    bias_z = stationary_gyro["gyro_z"].median()

    df["gyro_x_calibrated"] = df["gyro_x"] - bias_x
    df["gyro_y_calibrated"] = df["gyro_y"] - bias_y
    df["gyro_z_calibrated"] = df["gyro_z"] - bias_z

    bias_summary = {
        "stationary_sample_count": int(len(stationary_gyro)),
        "total_sample_count": int(len(df)),
        "stationary_fraction": float(len(stationary_gyro) / len(df)),
        "gyro_bias_rad_per_sec": {
            "x": bias_x,
            "y": bias_y,
            "z": bias_z,
        },
        "gyro_bias_deg_per_sec": {
            "x": float(np.degrees(bias_x)),
            "y": float(np.degrees(bias_y)),
            "z": float(np.degrees(bias_z)),
        },
        "stationary_gyro_std_rad_per_sec": {
            "x": float(stationary_gyro["gyro_x"].std()),
            "y": float(stationary_gyro["gyro_y"].std()),
            "z": float(stationary_gyro["gyro_z"].std()),
        },
    }
    return df,bias_summary

def compute_roll_pitch_from_gravity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ax = df["gravity_x"]
    ay = df["gravity_y"]
    az = df["gravity_z"]
    df["roll_rad"] = np.arctan2(ay, az)
    df["pitch_rad"] = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    df["roll_deg"] = np.degrees(df["roll_rad"])
    df["pitch_deg"] = np.degrees(df["pitch_rad"])
    return df
def integrate_gyro(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time = df["time_sec"].to_numpy()
    dt = np.diff(time, prepend=time[0])
    gyro_x = df["gyro_x_calibrated"].to_numpy()
    gyro_y = df["gyro_y_calibrated"].to_numpy()

    gyro_roll_rad = np.cumsum(gyro_x * dt)
    gyro_pitch_rad = np.cumsum(gyro_y * dt)

    gyro_roll_rad = df["roll_rad"].iloc[0] + gyro_roll_rad
    gyro_pitch_rad = df["pitch_rad"].iloc[0] + gyro_pitch_rad

    df["gyro_roll_rad"] = gyro_roll_rad
    df["gyro_pitch_rad"] = gyro_pitch_rad
    df["gyro_roll_deg"] = np.degrees(gyro_roll_rad)
    df["gyro_pitch_deg"] = np.degrees(gyro_pitch_rad)

    return df

def summarize_orientation(df: pd.DataFrame) -> dict:
    return {
        "roll_gravity_start_deg": float(df["roll_deg"].iloc[0]),
        "roll_gravity_end_deg": float(df["roll_deg"].iloc[-1]),
        "roll_gyro_start_deg": float(df["gyro_roll_deg"].iloc[0]),
        "roll_gyro_end_deg": float(df["gyro_roll_deg"].iloc[-1]),
        "pitch_gravity_start_deg": float(df["pitch_deg"].iloc[0]),
        "pitch_gravity_end_deg": float(df["pitch_deg"].iloc[-1]),
        "pitch_gyro_start_deg": float(df["gyro_pitch_deg"].iloc[0]),
        "pitch_gyro_end_deg": float(df["gyro_pitch_deg"].iloc[-1]),
        "roll_end_difference_deg": float(df["gyro_roll_deg"].iloc[-1] - df["roll_deg"].iloc[-1]),
        "pitch_end_difference_deg": float(df["gyro_pitch_deg"].iloc[-1] - df["pitch_deg"].iloc[-1]),
    }
def integrate_rate(time: np.ndarray, rate: np.ndarray, initial_angle: float) -> np.ndarray:
    dt = np.diff(time, prepend=time[0])
    angle_delta = np.cumsum(rate * dt)
    return initial_angle + angle_delta

def diagnose_gyro_axis_mapping(df: pd.DataFrame) -> dict:
    df = df.copy()
    time = df["time_sec"].to_numpy()

    candidates = {
        "+gyro_x": df["gyro_x_calibrated"].to_numpy(),
        "-gyro_x": -df["gyro_x_calibrated"].to_numpy(),
        "+gyro_y": df["gyro_y_calibrated"].to_numpy(),
        "-gyro_y": -df["gyro_y_calibrated"].to_numpy(),
    }
    roll_reference = np.unwrap(df["roll_rad"].to_numpy())
    pitch_reference = np.unwrap(df["pitch_rad"].to_numpy())

    roll_initial = roll_reference[0]
    pitch_initial = pitch_reference[0]

    roll_rmse = {}
    pitch_rmse = {}

    for name, rate in candidates.items():
        roll_candidate = integrate_rate(
            time=time,
            rate=rate,
            initial_angle=roll_initial,
        )

        pitch_candidate = integrate_rate(
            time=time,
            rate=rate,
            initial_angle=pitch_initial,
        )

        roll_error = roll_candidate - roll_reference
        pitch_error = pitch_candidate - pitch_reference

        roll_rmse[name] = float(np.degrees(np.sqrt(np.mean(roll_error**2))))
        pitch_rmse[name] = float(np.degrees(np.sqrt(np.mean(pitch_error**2))))

    best_roll_mapping = min(roll_rmse, key=roll_rmse.get)
    best_pitch_mapping = min(pitch_rmse, key=pitch_rmse.get)

    return {
        "roll_candidates_rmse_deg": roll_rmse,
        "pitch_candidates_rmse_deg": pitch_rmse,
        "best_roll_mapping": best_roll_mapping,
        "best_pitch_mapping": best_pitch_mapping,
    }

def summarize_angle_ranges(df: pd.DataFrame) -> dict:
    return {
        "roll_gravity_min_deg": float(df["roll_deg"].min()),
        "roll_gravity_max_deg": float(df["roll_deg"].max()),
        "roll_gravity_range_deg": float(df["roll_deg"].max() - df["roll_deg"].min()),
        "pitch_gravity_min_deg": float(df["pitch_deg"].min()),
        "pitch_gravity_max_deg": float(df["pitch_deg"].max()),
        "pitch_gravity_range_deg": float(df["pitch_deg"].max() - df["pitch_deg"].min()),
    }

def add_complementary_filter(df: pd.DataFrame, alpha: float = 0.98) -> pd.DataFrame:
    df = df.copy()
    time = df["time_sec"].to_numpy()
    dt = np.diff(time, prepend=time[0])

    gyro_x = df["gyro_x_calibrated"].to_numpy()
    gyro_y = df["gyro_y_calibrated"].to_numpy()

    gravity_roll = df["roll_rad"].to_numpy()
    gravity_pitch = df["pitch_rad"].to_numpy()

    fused_roll = np.zeros(len(df))
    fused_pitch = np.zeros(len(df))

    fused_roll[0] = gravity_roll[0]
    fused_pitch[0] = gravity_pitch[0]
    
    #Recursive complementary filter.
    for k in range(1, len(df)):
        # Gyro prediction step.
        roll_prediction = fused_roll[k - 1] + gyro_x[k] * dt[k]
        pitch_prediction = fused_pitch[k - 1] + gyro_y[k] * dt[k]

        # Gravity correction step.
        fused_roll[k] = (
            alpha * roll_prediction
            + (1.0 - alpha) * gravity_roll[k]
        )

        fused_pitch[k] = (
            alpha * pitch_prediction
            + (1.0 - alpha) * gravity_pitch[k]
        )

    df["fused_roll_rad"] = fused_roll
    df["fused_pitch_rad"] = fused_pitch

    df["fused_roll_deg"] = np.degrees(fused_roll)
    df["fused_pitch_deg"] = np.degrees(fused_pitch)

    return df
def summarize_filter_performance(df: pd.DataFrame) -> dict:
    """
    Compare gravity reference, gyro-only integration, and fused estimate.

    This treats gravity-derived roll/pitch as the long-term reference.
    That is not perfect truth, but it is useful for this controlled tilt test.
    """

    roll_gravity = df["roll_deg"].to_numpy()
    pitch_gravity = df["pitch_deg"].to_numpy()

    roll_gyro = df["gyro_roll_deg"].to_numpy()
    pitch_gyro = df["gyro_pitch_deg"].to_numpy()

    roll_fused = df["fused_roll_deg"].to_numpy()
    pitch_fused = df["fused_pitch_deg"].to_numpy()

    roll_gyro_error = roll_gyro - roll_gravity
    pitch_gyro_error = pitch_gyro - pitch_gravity

    roll_fused_error = roll_fused - roll_gravity
    pitch_fused_error = pitch_fused - pitch_gravity

    return {
        "roll_gravity_start_deg": float(roll_gravity[0]),
        "roll_gravity_end_deg": float(roll_gravity[-1]),
        "roll_gyro_end_deg": float(roll_gyro[-1]),
        "roll_fused_end_deg": float(roll_fused[-1]),

        "pitch_gravity_start_deg": float(pitch_gravity[0]),
        "pitch_gravity_end_deg": float(pitch_gravity[-1]),
        "pitch_gyro_end_deg": float(pitch_gyro[-1]),
        "pitch_fused_end_deg": float(pitch_fused[-1]),

        "roll_gyro_rmse_deg": float(np.sqrt(np.mean(roll_gyro_error ** 2))),
        "roll_fused_rmse_deg": float(np.sqrt(np.mean(roll_fused_error ** 2))),

        "pitch_gyro_rmse_deg": float(np.sqrt(np.mean(pitch_gyro_error ** 2))),
        "pitch_fused_rmse_deg": float(np.sqrt(np.mean(pitch_fused_error ** 2))),
    }
def save_orientation_plots(df: pd.DataFrame, output_dir: Path, experiment_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    time = df["time_sec"]

    # Plot x-axis tilt / roll
    plt.figure(figsize=(12, 5))
    plt.plot(time, df["roll_deg"], label="gravity roll")
    plt.plot(time, df["gyro_roll_deg"], label="gyro-only roll")
    plt.plot(time, df["fused_roll_deg"], label="fused roll")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(f"{experiment_id}: Roll / X-axis Tilt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_id}_roll_comparison.png")
    plt.close()

    # Plot y-axis tilt / pitch
    plt.figure(figsize=(12, 5))
    plt.plot(time, df["pitch_deg"], label="gravity pitch")
    plt.plot(time, df["gyro_pitch_deg"], label="gyro-only pitch")
    plt.plot(time, df["fused_pitch_deg"], label="fused pitch")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(f"{experiment_id}: Pitch / Y-axis Tilt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_id}_pitch_comparison.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=Path, default="/mnt/d/PhoneTelemetry/db/telemetry.db", help="Path to the SQLite database")
    parser.add_argument("--run_id_acc", type=str, required=True, help="Run ID to analyze for accelerometer data")
    parser.add_argument("--run_id_gyro", type=str, required=True, help="Run ID to analyze for gyroscope data")
    parser.add_argument("--run_id_gravity", type=str, required=True, help="Run ID to analyze for gravity data")
    parser.add_argument("--output_dir", type=Path, default=Path("./output"), help="Directory to save output files")
    parser.add_argument("--experiment_id", type=str, default="IMU_tilt_test", help="Name used for saved output files")
    args = parser.parse_args()
    #load data
    acc_df = load_measurements(args.db_path, args.run_id_acc)
    gyro_df = load_measurements(args.db_path, args.run_id_gyro)
    gravity_df = load_measurements(args.db_path, args.run_id_gravity)
    #pivot and rename
    wide_acc_df = pivot_sensor_data(acc_df)
    wide_gyro_df = pivot_sensor_data(gyro_df)
    wide_gravity_df = pivot_sensor_data(gravity_df)
    wide_acc_df = normalize_time(wide_acc_df)
    wide_gyro_df = normalize_time(wide_gyro_df)
    wide_gravity_df = normalize_time(wide_gravity_df)
    wide_acc_df = rename_columns(wide_acc_df, "acc")
    wide_gyro_df = rename_columns(wide_gyro_df, "gyro")
    wide_gravity_df = rename_columns(wide_gravity_df, "gravity")

    #align and merge
    aligned_df = align_and_merge(wide_acc_df, wide_gyro_df, wide_gravity_df)
    summary = summarize_aligned_data(aligned_df)

    #add magnitude columns
    aligned_df = add_magnitude_columns(aligned_df)

    #detect stationary periods
    aligned_df = detect_stationary_periods(aligned_df)
    #calibrate gyro bias
    aligned_df,bias_summary = calibrate_gyro_bias(aligned_df)
    print(json.dumps(bias_summary, indent=2))
    #compute roll and pitch from gravity vector
    aligned_df = compute_roll_pitch_from_gravity(aligned_df)
    #diagnosing correct gyro axis to integrate
    axis_mapping_summary = diagnose_gyro_axis_mapping(aligned_df)
    print(json.dumps(axis_mapping_summary, indent=2))
    #integrate gyro roll and pitch
    aligned_df = integrate_gyro(aligned_df)
    #complementary filter
    aligned_df = add_complementary_filter(aligned_df, alpha =0.98)

    filter_summary = summarize_filter_performance(aligned_df)
    print(json.dumps(filter_summary, indent=2))

    save_orientation_plots(aligned_df, output_dir =args.output_dir, experiment_id= args.experiment_id)


if __name__ == "__main__":
    main()