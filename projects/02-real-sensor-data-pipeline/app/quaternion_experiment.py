import pandas as pd
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

EXPERIMENT_ID = "imu_tilt_test"
EXPERIMENT_OUTPUT_DIR = OUTPUT_ROOT / EXPERIMENT_ID
PLOTS_DIR = EXPERIMENT_OUTPUT_DIR / "plots"
CSV_DIR = EXPERIMENT_OUTPUT_DIR / "csv"
SUMMARY_DIR = EXPERIMENT_OUTPUT_DIR / "summaries"

INPUT_CSV = PROCESSED_DATA_DIR / "IMU_tilt_test_data.csv"

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

def add_world_frame_acceleration(df: pd.DataFrame, use_transpose: bool = False) -> pd.DataFrame:
    df = df.copy()

    quats = df[["q_w", "q_x", "q_y", "q_z"]].to_numpy()
    acc_phone = df[["acc_x", "acc_y", "acc_z"]].to_numpy()

    acc_world = np.zeros_like(acc_phone)

    for k in range(len(df)):
        R = quat_to_rotation_matrix(quats[k])
        if use_transpose:
            acc_world[k] = R.T @ acc_phone[k]
        else:
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

def add_motion_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_label"] = "unknown"
    
    lin_acc = (
        df["linear_acc_world_mag"]
        .rolling(window=10, center=True, min_periods=1)
        .mean()
        )
    gyro = (
        df["gyro_mag"]
        .rolling(window=10, center=True, min_periods=1)
        .mean()
    )
    df["linear_acc_world_mag_smooth"] = lin_acc
    df["gyro_mag_smooth"] = gyro

    stationary = (lin_acc < 0.35) & (gyro < 0.05)
    small_motion = (lin_acc >= 0.35) & (lin_acc < 1.0) & (gyro < 0.05)
    rotation = (gyro >= 0.05) & (lin_acc < 1.0)
    motion = (lin_acc >= 1.0) & (lin_acc < 4.0)
    high_accel = (lin_acc >= 4.0) & (lin_acc < 8.0)
    impact = lin_acc >= 8.0

    df.loc[stationary, "event_label"] = "stationary"
    df.loc[small_motion, "event_label"] = "small_motion"
    df.loc[rotation, "event_label"] = "rotation"
    df.loc[motion, "event_label"] = "motion"
    df.loc[high_accel, "event_label"] = "high_accel"
    df.loc[impact, "event_label"] = "impact_or_shake"

    return df

def add_event_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    group_map = {
        "stationary": "stationary",
        "small_motion": "small_motion",
        "rotation": "rotation",
        "motion": "active_motion",
        "high_accel": "active_motion",
        "impact_or_shake": "impact_or_shake",
        "unknown": "unknown",
    }

    df["event_group"] = df["event_label"].map(group_map).fillna("unknown")
    return df

def merge_tiny_label_segments(
    df: pd.DataFrame,
    label_col: str,
    time_col: str = "time_sec",
    min_duration_sec: float = 0.15,
    protected_labels: set[str] | None = None,
    max_passes: int = 5,
) -> pd.DataFrame:
    df = df.copy()

    if protected_labels is None:
        protected_labels = {"impact_or_shake"}

    mean_dt = df[time_col].diff().median()

    for _ in range(max_passes):
        segment_col = "_tmp_segment_id"

        df[segment_col] = (
            df[label_col] != df[label_col].shift()
        ).cumsum()

        segments = (
            df.reset_index()
            .groupby(segment_col)
            .agg(
                label=(label_col, "first"),
                start_idx=("index", "first"),
                end_idx=("index", "last"),
                start_time_sec=(time_col, "first"),
                end_time_sec=(time_col, "last"),
                sample_count=(time_col, "count"),
            )
        )

        segments["duration_sec"] = (
            segments["end_time_sec"] - segments["start_time_sec"] + mean_dt
        )

        tiny_segments = segments[
            (segments["duration_sec"] < min_duration_sec)
            & (~segments["label"].isin(protected_labels))
        ]

        if tiny_segments.empty:
            break

        changed = False

        for segment_id, segment in tiny_segments.iterrows():
            prev_id = segment_id - 1
            next_id = segment_id + 1

            has_prev = prev_id in segments.index
            has_next = next_id in segments.index

            if not has_prev and not has_next:
                continue

            if has_prev and has_next:
                prev_label = segments.loc[prev_id, "label"]
                next_label = segments.loc[next_id, "label"]

                prev_duration = segments.loc[prev_id, "duration_sec"]
                next_duration = segments.loc[next_id, "duration_sec"]

                # If both neighbors agree, merge into that label.
                if prev_label == next_label:
                    replacement_label = prev_label
                # Otherwise merge into the longer neighbor.
                elif prev_duration >= next_duration:
                    replacement_label = prev_label
                else:
                    replacement_label = next_label

            elif has_prev:
                replacement_label = segments.loc[prev_id, "label"]
            else:
                replacement_label = segments.loc[next_id, "label"]

            start_idx = segment["start_idx"]
            end_idx = segment["end_idx"]

            df.loc[start_idx:end_idx, label_col] = replacement_label
            changed = True

        df = df.drop(columns=[segment_col])

        if not changed:
            break

    if "_tmp_segment_id" in df.columns:
        df = df.drop(columns=["_tmp_segment_id"])

    return df

def summarize_event_segments(
    df: pd.DataFrame,
    label_col: str = "event_label",
) -> pd.DataFrame:
    df = df.copy()

    segment_col = f"{label_col}_segment_id"

    df[segment_col] = (
        df[label_col] != df[label_col].shift()
    ).cumsum()

    segments = (
        df.groupby([segment_col, label_col])
        .agg(
            start_time_sec=("time_sec", "first"),
            end_time_sec=("time_sec", "last"),
            sample_count=("time_sec", "count"),
            max_linear_acc_mps2=("linear_acc_world_mag", "max"),
            mean_linear_acc_mps2=("linear_acc_world_mag", "mean"),
            max_linear_acc_smooth_mps2=("linear_acc_world_mag_smooth", "max"),
            mean_linear_acc_smooth_mps2=("linear_acc_world_mag_smooth", "mean"),
            max_gyro_mag_rad_s=("gyro_mag", "max"),
            mean_gyro_mag_rad_s=("gyro_mag", "mean"),
            max_gyro_mag_smooth_rad_s=("gyro_mag_smooth", "max"),
            mean_gyro_mag_smooth_rad_s=("gyro_mag_smooth", "mean"),
        )
        .reset_index()
    )

    mean_dt = df["time_sec"].diff().median()

    segments["duration_sec"] = (
        segments["end_time_sec"] - segments["start_time_sec"] + mean_dt
    )

    segments["is_tiny_segment"] = segments["duration_sec"] < 0.15

    return segments

def summarize_motion_events(df: pd.DataFrame) -> dict:
    counts = df["event_label"].value_counts().to_dict()
    percentages = (df["event_label"].value_counts(normalize=True) * 100).to_dict()

    return {
        "event_counts": {k: int(v) for k, v in counts.items()},
        "event_percentages": {k: float(v) for k, v in percentages.items()},
    }

def summarize_linear_acceleration(df: pd.DataFrame) -> dict:
    lin_mag = df["linear_acc_world_mag"].to_numpy()

    return {
        "linear_acc_world_mag_mean_mps2": float(np.mean(lin_mag)),
        "linear_acc_world_mag_std_mps2": float(np.std(lin_mag)),
        "linear_acc_world_mag_max_mps2": float(np.max(lin_mag)),
        "linear_acc_world_mag_95th_percentile_mps2": float(np.percentile(lin_mag, 95)),
        "linear_acc_world_z_mean_mps2": float(df["linear_acc_world_z"].mean()),
    }

def save_linear_acceleration_plot(
    df: pd.DataFrame,
    output_dir: Path,
    experiment_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    time = df["time_sec"]

    plt.figure(figsize=(12, 5))
    plt.plot(time, df["linear_acc_world_mag"], label="linear acceleration magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("m/s²")
    plt.title(f"{experiment_id}: Gravity-Removed Linear Acceleration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_id}_linear_acc_world_mag.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(time, df["acc_world_z"], label="world z acceleration")
    plt.plot(time, df["linear_acc_world_z"], label="world z after gravity removal")
    plt.xlabel("Time (s)")
    plt.ylabel("m/s²")
    plt.title(f"{experiment_id}: World Z Acceleration Before/After Gravity Removal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_id}_world_z_gravity_removal.png")
    plt.close()

def save_motion_event_threshold_plot(
    df: pd.DataFrame,
    output_dir: Path,
    experiment_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    time = df["time_sec"]

    plt.figure(figsize=(12, 5))

    plt.plot(
        time,
        df["linear_acc_world_mag_smooth"],
        label="smoothed linear acceleration magnitude",
        color="tab:blue",
        linewidth=1.5,
    )

    thresholds = [
        (0.35, "stationary / small motion"),
        (1.0, "motion"),
        (4.0, "high acceleration"),
        (8.0, "impact / shake"),
    ]

    for value, label in thresholds:
        plt.axhline(
            value,
            linestyle="--",
            linewidth=1.0,
            color="gray",
            alpha=0.8,
            label=label,
        )

    plt.xlabel("Time (s)")
    plt.ylabel("m/s²")
    plt.title(f"{experiment_id}: Motion Event Thresholds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_id}_motion_event_thresholds.png")
    plt.close()

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_run_summary(
    experiment_id: str,
    norm_summary: dict,
    quat_vs_euler_summary: dict,
    quat_vs_gravity_summary: dict,
    acc_summary: dict,
    event_summary: dict,
    fine_segments: pd.DataFrame,
    group_segments: pd.DataFrame,
    clean_group_segments: pd.DataFrame,
) -> dict:
    return {
        "experiment_id": experiment_id,
        "sample_count": int(
            event_summary["event_counts"]
            and sum(event_summary["event_counts"].values())
        ),
        "quaternion_norm": norm_summary,
        "quaternion_vs_euler_gyro": quat_vs_euler_summary,
        "quaternion_vs_gravity": quat_vs_gravity_summary,
        "linear_acceleration": acc_summary,
        "motion_events": event_summary,
        "segment_cleanup": {
            "fine_label_segments": int(len(fine_segments)),
            "fine_tiny_segments": int(fine_segments["is_tiny_segment"].sum()),
            "group_segments": int(len(group_segments)),
            "group_tiny_segments": int(group_segments["is_tiny_segment"].sum()),
            "clean_group_segments": int(len(clean_group_segments)),
            "clean_group_tiny_segments": int(clean_group_segments["is_tiny_segment"].sum()),
        },
    }


def _display_path(path: Path) -> str:
    """Return a project-relative path when possible, otherwise a normal string."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def save_experiment_outputs(
    df: pd.DataFrame,
    fine_segments: pd.DataFrame,
    group_segments: pd.DataFrame,
    clean_group_segments: pd.DataFrame,
    norm_summary: dict,
    quat_vs_euler_summary: dict,
    quat_vs_gravity_summary: dict,
    acc_summary: dict,
    event_summary: dict,
    experiment_id: str,
    csv_dir: Path = CSV_DIR,
    summary_dir: Path = SUMMARY_DIR,
) -> dict:
    csv_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    run_summary = build_run_summary(
        experiment_id=experiment_id,
        norm_summary=norm_summary,
        quat_vs_euler_summary=quat_vs_euler_summary,
        quat_vs_gravity_summary=quat_vs_gravity_summary,
        acc_summary=acc_summary,
        event_summary=event_summary,
        fine_segments=fine_segments,
        group_segments=group_segments,
        clean_group_segments=clean_group_segments,
    )

    output_paths = {
        "processed_events_csv": csv_dir / f"{experiment_id}_world_acc_events.csv",
        "fine_segments_csv": csv_dir / f"{experiment_id}_fine_event_segments.csv",
        "group_segments_csv": csv_dir / f"{experiment_id}_group_event_segments.csv",
        "clean_group_segments_csv": csv_dir / f"{experiment_id}_clean_group_segments.csv",
        "run_summary_json": summary_dir / f"{experiment_id}_run_summary.json",
        "quat_norm_summary_json": summary_dir / f"{experiment_id}_quat_norm_summary.json",
        "quat_vs_euler_summary_json": summary_dir / f"{experiment_id}_quat_vs_euler_summary.json",
        "quat_vs_gravity_summary_json": summary_dir / f"{experiment_id}_quat_vs_gravity_summary.json",
        "linear_acc_summary_json": summary_dir / f"{experiment_id}_linear_acc_summary.json",
        "event_summary_json": summary_dir / f"{experiment_id}_event_summary.json",
    }

    df.to_csv(output_paths["processed_events_csv"], index=False)
    fine_segments.to_csv(output_paths["fine_segments_csv"], index=False)
    group_segments.to_csv(output_paths["group_segments_csv"], index=False)
    clean_group_segments.to_csv(output_paths["clean_group_segments_csv"], index=False)

    save_json(run_summary, output_paths["run_summary_json"])
    save_json(norm_summary, output_paths["quat_norm_summary_json"])
    save_json(quat_vs_euler_summary, output_paths["quat_vs_euler_summary_json"])
    save_json(quat_vs_gravity_summary, output_paths["quat_vs_gravity_summary_json"])
    save_json(acc_summary, output_paths["linear_acc_summary_json"])
    save_json(event_summary, output_paths["event_summary_json"])

    return {
        name: _display_path(path)
        for name, path in output_paths.items()
    }

def main():
    for directory in (PLOTS_DIR, CSV_DIR, SUMMARY_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Expected processed input CSV at {INPUT_CSV}. "
            "Move or generate IMU_tilt_test_data.csv under data/processed/."
        )

    df = pd.read_csv(INPUT_CSV)

    initial_q = euler_to_quat(
        roll=df["roll_rad"].iloc[0],
        pitch=df["pitch_rad"].iloc[0],
        yaw=0.0,
    )

    # Orientation estimation
    df = integrate_quaternion_gyro(df, initial_quat=initial_q)
    df = add_quaternion_euler_columns(df)
    df = add_relative_orientation_columns(df)
    df = add_gravity_delta_columns(df)

    norm_summary = summarize_quat_norm(df)
    quat_vs_euler_summary = summarize_quaternion_vs_euler_gyro(df)
    quat_vs_gravity_summary = summarize_quaternion_vs_gravity(df)

    # World-frame acceleration and event labeling
    df = add_world_frame_acceleration(df)
    df = add_motion_event_labels(df)
    df = add_event_groups(df)

    df["event_group_clean"] = df["event_group"]
    df = merge_tiny_label_segments(
        df,
        label_col="event_group_clean",
        min_duration_sec=0.15,
    )

    fine_segments = summarize_event_segments(df, label_col="event_label")
    group_segments = summarize_event_segments(df, label_col="event_group")
    clean_group_segments = summarize_event_segments(df, label_col="event_group_clean")

    acc_summary = summarize_linear_acceleration(df)
    event_summary = summarize_motion_events(df)

    output_paths = save_experiment_outputs(
        df=df,
        fine_segments=fine_segments,
        group_segments=group_segments,
        clean_group_segments=clean_group_segments,
        norm_summary=norm_summary,
        quat_vs_euler_summary=quat_vs_euler_summary,
        quat_vs_gravity_summary=quat_vs_gravity_summary,
        acc_summary=acc_summary,
        event_summary=event_summary,
        experiment_id=EXPERIMENT_ID,
        csv_dir=CSV_DIR,
        summary_dir=SUMMARY_DIR,
    )

    save_quaternion_comparison_plots(
        df,
        output_dir=PLOTS_DIR,
        experiment_id=EXPERIMENT_ID,
    )
    save_linear_acceleration_plot(
        df,
        output_dir=PLOTS_DIR,
        experiment_id=EXPERIMENT_ID,
    )
    save_motion_event_threshold_plot(
        df,
        output_dir=PLOTS_DIR,
        experiment_id=EXPERIMENT_ID,
    )

    print(json.dumps(norm_summary, indent=2))
    print(json.dumps(quat_vs_euler_summary, indent=2))
    print(json.dumps(quat_vs_gravity_summary, indent=2))
    print("Fine label segments:", len(fine_segments))
    print("Fine tiny segments:", int(fine_segments["is_tiny_segment"].sum()))
    print("Group segments:", len(group_segments))
    print("Group tiny segments:", int(group_segments["is_tiny_segment"].sum()))
    print("Clean group segments:", len(clean_group_segments))
    print("Clean group tiny segments:", int(clean_group_segments["is_tiny_segment"].sum()))
    print(clean_group_segments.head(30).to_string(index=False))
    print(json.dumps(acc_summary, indent=2))
    print(json.dumps(event_summary, indent=2))
    print(json.dumps(output_paths, indent=2))


if __name__ == "__main__":
    main()
