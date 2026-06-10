import pandas as pd
import sqlite3
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt



def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else q

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def axis_angle_to_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("Rotation axis cannot be the zero vector.")

    axis_unit = axis / axis_norm
    half_angle = angle_rad / 2.0
    w = np.cos(half_angle)
    xyz = axis_unit * np.sin(half_angle)
    q = np.concatenate(([w], xyz))
    return quat_normalize(q)

def gyro_to_quat_delta(omega: np.ndarray, dt: float) -> np.ndarray:
    omega = np.asarray(omega, dtype=float)
    rotation_vector = omega * dt
    angle = np.linalg.norm(rotation_vector)
    if angle == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    axis = rotation_vector / angle
    return axis_angle_to_quat(axis, angle)

def integrate_quaternion_gyro(df: pd.DataFrame, initial_quat: np.ndarray | None = None) -> pd.DataFrame:
    df = df.copy()

    time = df["time_sec"].to_numpy()
    dt = np.diff(time, prepend=time[0])

    gyro = df[["gyro_x_calibrated", "gyro_y_calibrated", "gyro_z_calibrated"]].to_numpy()
    if initial_quat is None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        q = quat_normalize(initial_quat)
    quats = np.zeros((len(df), 4))
    quats[0] = q

    for k in range(1, len(df)):
        omega = gyro[k]
        delta_q = gyro_to_quat_delta(omega, dt[k])
        q = quat_multiply(q, delta_q)
        q = quat_normalize(q)
        quats[k] = q
    
    df["q_w"] = quats[:, 0]
    df["q_x"] = quats[:, 1]
    df["q_y"] = quats[:, 2]
    df["q_z"] = quats[:, 3]
    return df

def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = q
    # Roll around x-axis
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch around y-axis
    sinp = 2.0 * (w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw around z-axis
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return quat_normalize(np.array([w, x, y, z]))


def add_quaternion_euler_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    quats = df[["q_w", "q_x", "q_y", "q_z"]].to_numpy()

    euler = np.array([quat_to_euler(q) for q in quats])

    df["quat_roll_rad"] = euler[:, 0]
    df["quat_pitch_rad"] = euler[:, 1]
    df["quat_yaw_rad"] = euler[:, 2]

    df["quat_roll_deg"] = np.degrees(df["quat_roll_rad"])
    df["quat_pitch_deg"] = np.degrees(df["quat_pitch_rad"])
    df["quat_yaw_deg"] = np.degrees(df["quat_yaw_rad"])

    return df
def summarize_quat_norm(df: pd.DataFrame) -> dict:
    q = df[["q_w", "q_x", "q_y", "q_z"]].to_numpy()
    q_norm = np.linalg.norm(q, axis=1)
    return {
        "quat_norm_min": float(q_norm.min()),
        "quat_norm_max": float(q_norm.max()),
        "quat_norm_mean": float(q_norm.mean()),
        "quat_norm_std": float(q_norm.std()),
        "quat_norm_max_error_from_1": float(np.max(np.abs(q_norm - 1.0))),
    }
def add_relative_orientation_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gyro_roll_delta_deg"] = df["gyro_roll_deg"] - df["gyro_roll_deg"].iloc[0]
    df["gyro_pitch_delta_deg"] = df["gyro_pitch_deg"] - df["gyro_pitch_deg"].iloc[0]

    df["quat_roll_delta_deg"] = df["quat_roll_deg"] - df["quat_roll_deg"].iloc[0]
    df["quat_pitch_delta_deg"] = df["quat_pitch_deg"] - df["quat_pitch_deg"].iloc[0]
    df["quat_yaw_delta_deg"] = df["quat_yaw_deg"] - df["quat_yaw_deg"].iloc[0]

    return df

def add_gravity_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["gravity_roll_delta_deg"] = df["roll_deg"] - df["roll_deg"].iloc[0]
    df["gravity_pitch_delta_deg"] = df["pitch_deg"] - df["pitch_deg"].iloc[0]
    return df


def save_quaternion_comparison_plots(
    df: pd.DataFrame,
    output_dir: Path,
    experiment_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    time = df["time_sec"]

    # Roll / x-axis comparison
    plt.figure(figsize=(12, 5))
    plt.plot(time, df["gyro_roll_delta_deg"], label="Euler gyro roll delta")
    plt.plot(time, df["quat_roll_delta_deg"], label="Quaternion roll delta")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle change (deg)")
    plt.title(f"{experiment_id}: Euler Gyro vs Quaternion Roll")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_id}_quat_vs_euler_roll.png")
    plt.close()

    # Pitch / y-axis comparison
    plt.figure(figsize=(12, 5))
    plt.plot(time, df["gyro_pitch_delta_deg"], label="Euler gyro pitch delta")
    plt.plot(time, df["quat_pitch_delta_deg"], label="Quaternion pitch delta")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle change (deg)")
    plt.title(f"{experiment_id}: Euler Gyro vs Quaternion Pitch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_id}_quat_vs_euler_pitch.png")
    plt.close()

    # Yaw from quaternion
    plt.figure(figsize=(12, 5))
    plt.plot(time, df["quat_yaw_delta_deg"], label="Quaternion yaw delta")
    plt.xlabel("Time (s)")
    plt.ylabel("Yaw change (deg)")
    plt.title(f"{experiment_id}: Quaternion Yaw Drift / Rotation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_id}_quat_yaw.png")
    plt.close()

def summarize_quaternion_vs_euler_gyro(df: pd.DataFrame) -> dict:
    roll_error = df["quat_roll_delta_deg"] - df["gyro_roll_delta_deg"]
    pitch_error = df["quat_pitch_delta_deg"] - df["gyro_pitch_delta_deg"]

    return {
        "quat_vs_euler_roll_rmse_deg": float(np.sqrt(np.mean(roll_error**2))),
        "quat_vs_euler_pitch_rmse_deg": float(np.sqrt(np.mean(pitch_error**2))),
        "quat_roll_delta_end_deg": float(df["quat_roll_delta_deg"].iloc[-1]),
        "euler_roll_delta_end_deg": float(df["gyro_roll_delta_deg"].iloc[-1]),
        "quat_pitch_delta_end_deg": float(df["quat_pitch_delta_deg"].iloc[-1]),
        "euler_pitch_delta_end_deg": float(df["gyro_pitch_delta_deg"].iloc[-1]),
        "quat_yaw_delta_end_deg": float(df["quat_yaw_delta_deg"].iloc[-1]),
    }

def summarize_quaternion_vs_gravity(df: pd.DataFrame) -> dict:
    roll_error = df["quat_roll_delta_deg"] - df["gravity_roll_delta_deg"]
    pitch_error = df["quat_pitch_delta_deg"] - df["gravity_pitch_delta_deg"]
    
    return {
        "quat_vs_gravity_roll_rmse_deg": float(np.sqrt(np.mean(roll_error**2))),
        "quat_vs_gravity_pitch_rmse_deg": float(np.sqrt(np.mean(pitch_error**2))),
        "quat_roll_delta_end_deg": float(df["quat_roll_delta_deg"].iloc[-1]),
        "gravity_roll_delta_end_deg": float(df["gravity_roll_delta_deg"].iloc[-1]),
        "quat_pitch_delta_end_deg": float(df["quat_pitch_delta_deg"].iloc[-1]),
        "gravity_pitch_delta_end_deg": float(df["gravity_pitch_delta_deg"].iloc[-1]),
    }

def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [    2*(x*y + z*w),   1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x**2 + y**2)]
    ])

def add_world_frame_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    quats = df[["q_w", "q_x", "q_y", "q_z"]].to_numpy()
    acc_phone = df[["acc_x", "acc_y", "acc_z"]].to_numpy()

    acc_world = np.zeros_like(acc_phone)

    for k in range(len(df)):
        R = quat_to_rotation_matrix(quats[k])
        acc_world[k] = R @ acc_phone[k]

    df["acc_world_x"] = acc_world[:, 0]
    df["acc_world_y"] = acc_world[:, 1]
    df["acc_world_z"] = acc_world[:, 2]

    df["linear_acc_world_x"] = df["acc_world_x"]
    df["linear_acc_world_y"] = df["acc_world_y"]
    df["linear_acc_world_z"] = df["acc_world_z"] - 9.80665

    df["linear_acc_world_mag"] = np.sqrt(
        df["linear_acc_world_x"]**2
        + df["linear_acc_world_y"]**2
        + df["linear_acc_world_z"]**2
    )

    return df

def summarize_linear_acceleration(df: pd.DataFrame) -> dict:
    lin_mag = df["linear_acc_world_mag"].to_numpy()

    return {
        "linear_acc_world_mag_mean_mps2": float(np.mean(lin_mag)),
        "linear_acc_world_mag_std_mps2": float(np.std(lin_mag)),
        "linear_acc_world_mag_max_mps2": float(np.max(lin_mag)),
        "linear_acc_world_mag_95th_percentile_mps2": float(np.percentile(lin_mag, 95)),
        "linear_acc_world_z_mean_mps2": float(df["linear_acc_world_z"].mean()),
    }



def main():
    df = pd.read_csv("output/IMU_tilt_test_data.csv")
    initial_q = euler_to_quat(
        roll = df["roll_rad"].iloc[0],
        pitch = df["pitch_rad"].iloc[0],
        yaw = 0.0,
    )
    df = integrate_quaternion_gyro(df, initial_quat=initial_q)
    df = add_quaternion_euler_columns(df)
    df = add_relative_orientation_columns(df)
    df = add_gravity_delta_columns(df)
    norm_summary = summarize_quat_norm(df)
    print(json.dumps(norm_summary, indent=2))
    save_quaternion_comparison_plots(df, output_dir=Path("output"), experiment_id="imu_tilt_test")
    summary = summarize_quaternion_vs_euler_gyro(df)
    print(json.dumps(summary, indent=2))
    gravity_summary = summarize_quaternion_vs_gravity(df)
    print(json.dumps(gravity_summary, indent=2))
    df = add_world_frame_acceleration(df)
    acc_summary = summarize_linear_acceleration(df)
    print(json.dumps(acc_summary, indent=2))

if __name__ == "__main__":
    main()