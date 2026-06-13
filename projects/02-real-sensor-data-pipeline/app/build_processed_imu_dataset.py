from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one aligned processed IMU dataset from raw phyphox CSV files."
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID, e.g. imu_labeled_motion_001",
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Folder containing raw accelerometer/gravity/gyroscope CSVs.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output processed CSV path. Default: data/processed/<run-id>.csv",
    )

    parser.add_argument(
        "--bias-window-sec",
        type=float,
        default=3.0,
        help="Initial stationary window used to estimate gyro bias.",
    )

    return parser.parse_args()


def find_one_file(raw_dir: Path, pattern: str) -> Path:
    matches = sorted(raw_dir.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for pattern: {raw_dir / pattern}")

    if len(matches) > 1:
        raise RuntimeError(
            f"Expected one file for pattern {pattern}, found {len(matches)}:\n"
            + "\n".join(str(p) for p in matches)
        )

    return matches[0]


def load_raw_frames(raw_dir: Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acc_path = find_one_file(raw_dir, f"{run_id}_accelerometer.csv")
    gravity_path = find_one_file(raw_dir, f"{run_id}_gravity.csv")
    gyro_path = find_one_file(raw_dir, f"{run_id}_gyroscope.csv")

    acc = pd.read_csv(acc_path)
    gravity = pd.read_csv(gravity_path)
    gyro = pd.read_csv(gyro_path)

    return acc, gravity, gyro


def reset_time_zero(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    min_time = min(float(frame["time_sec"].iloc[0]) for frame in frames)

    shifted = []
    for frame in frames:
        frame = frame.copy()
        frame["time_sec"] = frame["time_sec"] - min_time
        shifted.append(frame)

    return shifted


def interp_to_time(
    source_time: np.ndarray,
    source_values: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    return np.interp(target_time, source_time, source_values)


def build_aligned_dataset(
    acc: pd.DataFrame,
    gravity: pd.DataFrame,
    gyro: pd.DataFrame,
    bias_window_sec: float,
) -> pd.DataFrame:
    acc, gravity, gyro = reset_time_zero(acc, gravity, gyro)

    # Use gyro timestamps as the master timeline.
    t = gyro["time_sec"].to_numpy()

    df = pd.DataFrame()
    df["time_sec"] = t

    # Gyro already lives on master timeline.
    df["gyro_x"] = gyro["gyro_x"].to_numpy()
    df["gyro_y"] = gyro["gyro_y"].to_numpy()
    df["gyro_z"] = gyro["gyro_z"].to_numpy()

    # Interpolate accelerometer onto gyro time.
    for col in ["acc_x", "acc_y", "acc_z"]:
        df[col] = interp_to_time(
            acc["time_sec"].to_numpy(),
            acc[col].to_numpy(),
            t,
        )

    # Interpolate gravity onto gyro time.
    for col in ["gravity_x", "gravity_y", "gravity_z"]:
        df[col] = interp_to_time(
            gravity["time_sec"].to_numpy(),
            gravity[col].to_numpy(),
            t,
        )

    # Magnitudes.
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["gravity_mag"] = np.sqrt(
        df["gravity_x"]**2 + df["gravity_y"]**2 + df["gravity_z"]**2
    )

    # Estimate gyro bias from the first stationary window.
    bias_mask = df["time_sec"] <= bias_window_sec

    if bias_mask.sum() < 5:
        raise RuntimeError(
            f"Not enough samples in bias window: {bias_mask.sum()} samples"
        )

    gyro_bias_x = float(df.loc[bias_mask, "gyro_x"].mean())
    gyro_bias_y = float(df.loc[bias_mask, "gyro_y"].mean())
    gyro_bias_z = float(df.loc[bias_mask, "gyro_z"].mean())

    df["gyro_x_calibrated"] = df["gyro_x"] - gyro_bias_x
    df["gyro_y_calibrated"] = df["gyro_y"] - gyro_bias_y
    df["gyro_z_calibrated"] = df["gyro_z"] - gyro_bias_z

    df["gyro_mag_raw"] = np.sqrt(
        df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2
    )

    df["gyro_mag"] = np.sqrt(
        df["gyro_x_calibrated"]**2
        + df["gyro_y_calibrated"]**2
        + df["gyro_z_calibrated"]**2
    )

    # Gravity-based roll/pitch.
    df["roll_rad"] = np.arctan2(df["gravity_y"], df["gravity_z"])

    df["pitch_rad"] = np.arctan2(
        -df["gravity_x"],
        np.sqrt(df["gravity_y"]**2 + df["gravity_z"]**2),
    )

    df["roll_deg"] = np.degrees(df["roll_rad"])
    df["pitch_deg"] = np.degrees(df["pitch_rad"])

    # Simple gyro roll/pitch integration for comparison/debugging.
    dt = df["time_sec"].diff().fillna(0.0).to_numpy()

    df["gyro_roll_rad"] = (
        df["roll_rad"].iloc[0]
        + np.cumsum(df["gyro_x_calibrated"].to_numpy() * dt)
    )

    df["gyro_pitch_rad"] = (
        df["pitch_rad"].iloc[0]
        + np.cumsum(df["gyro_y_calibrated"].to_numpy() * dt)
    )

    df["gyro_roll_deg"] = np.degrees(df["gyro_roll_rad"])
    df["gyro_pitch_deg"] = np.degrees(df["gyro_pitch_rad"])

    # Store bias values as constant columns for traceability.
    df["gyro_bias_x"] = gyro_bias_x
    df["gyro_bias_y"] = gyro_bias_y
    df["gyro_bias_z"] = gyro_bias_z
    df["bias_window_sec"] = bias_window_sec

    return df


def print_summary(df: pd.DataFrame) -> None:
    duration = float(df["time_sec"].iloc[-1] - df["time_sec"].iloc[0])
    mean_dt = float(df["time_sec"].diff().median())
    sample_rate = 1.0 / mean_dt if mean_dt > 0 else float("nan")

    print("\nProcessed dataset summary:")
    print(f"rows:              {len(df)}")
    print(f"duration_sec:      {duration:.3f}")
    print(f"mean_dt_sec:       {mean_dt:.6f}")
    print(f"sample_rate_hz:    {sample_rate:.2f}")
    print(f"acc_mag_mean:      {df['acc_mag'].mean():.4f}")
    print(f"gravity_mag_mean:  {df['gravity_mag'].mean():.4f}")
    print(f"gyro_mag_mean:     {df['gyro_mag'].mean():.6f}")
    print("\nGyro bias:")
    print(f"x: {df['gyro_bias_x'].iloc[0]: .8f} rad/s")
    print(f"y: {df['gyro_bias_y'].iloc[0]: .8f} rad/s")
    print(f"z: {df['gyro_bias_z'].iloc[0]: .8f} rad/s")


def main() -> None:
    args = parse_args()

    raw_dir = args.raw_dir
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir

    output_path = args.output
    if output_path is None:
        output_path = PROCESSED_DIR / f"{args.run_id}.csv"
    elif not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    acc, gravity, gyro = load_raw_frames(raw_dir, args.run_id)

    df = build_aligned_dataset(
        acc=acc,
        gravity=gravity,
        gyro=gyro,
        bias_window_sec=args.bias_window_sec,
    )

    df.to_csv(output_path, index=False)

    print_summary(df)
    print(f"\nsaved: {output_path}")


if __name__ == "__main__":
    main()