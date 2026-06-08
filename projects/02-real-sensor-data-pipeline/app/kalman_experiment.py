import pandas as pd
import sqlite3
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def run_kalman_angle_bias(df: pd.DataFrame, gyro_col: str, measurement_col: str, initial_angle: float | None = None,
q_angle: float = 1e-5,
q_bias: float = 1e-7,
r_measurement: float = 1e-3,
 ) -> pd.DataFrame:
    df = df.copy()
    time = df["time_sec"].to_numpy()
    dt_array = np.diff(time, prepend=time[0])
    gyro_rate = df[gyro_col].to_numpy()
    z_angle = df[measurement_col].to_numpy()

    n = len(df)

    angle_est = np.zeros(n)
    bias_est = np.zeros(n)

    if initial_angle is None:
        initial_angle = z_angle[0]
    
    x = np.array([initial_angle, 0.0])

    P = np.eye(2)
    Q = np.array([
        [q_angle, 0.0],
        [0.0, q_bias],
    ])

    R = np.array([[r_measurement]])
    I = np.eye(2)
    H = np.array([[1.0, 0.0]])

    for k in range(n):
        dt = dt_array[k]
        F = np.array([
            [1.0, -dt],
            [0.0, 1.0],
        ])

        B = np.array([
            dt,
            0.0,
        ])
        u = gyro_rate[k]
        #Predict
        x_pred = F @ x + B * u
        P_pred = F @ P @ F.T + Q
        #update
        z = np.array([z_angle[k]])
        innovation = z - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + (K @ innovation)
        P = (I - K @ H) @ P_pred
        angle_est[k] = x[0]
        bias_est[k] = x[1]
    
    df["kalman_angle_rad"] = angle_est
    df["kalman_angle_deg"] = np.degrees(angle_est)
    df["kalman_bias_rad_s"] = bias_est
    df["kalman_bias_deg_s"] = np.degrees(bias_est)
    return df

def rmse_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a-b)**2)))

def sweep_kalman_parameters(df: pd.DataFrame,
    gyro_col: str,
    measurement_col: str,
    q_angle_values: list[float],
    q_bias_values: list[float],
    r_values: list[float],
    ) -> pd.DataFrame:
    results = []
    for q_angle in q_angle_values:
        for q_bias in q_bias_values:
            for r in r_values:
                df_kalman = run_kalman_angle_bias(
                    df,
                    gyro_col=gyro_col,
                    measurement_col=measurement_col,
                    q_angle=q_angle,
                    q_bias=q_bias,
                    r_measurement=r,
                )
                rmse = rmse_deg(df_kalman["kalman_angle_deg"].to_numpy(), np.degrees(df[measurement_col].to_numpy()))
                results.append({
                    "q_angle": q_angle,
                    "q_bias": q_bias,
                    "r_measurement": r,
                    "kalman_rmse_deg": rmse,
                    "bias_end_deg_s": float(df_kalman["kalman_bias_deg_s"].iloc[-1]),
                })
    sweep_df = pd.DataFrame(results)
    sweep_df = sweep_df.sort_values("kalman_rmse_deg").reset_index(drop=True)
    return sweep_df

def summarize_kalman_performance(df: pd.DataFrame) -> dict:
    gravity_pitch = df["pitch_deg"].to_numpy()
    gyro_pitch = df["gyro_pitch_deg"].to_numpy()
    fixed_pitch = df["fused_pitch_deg"].to_numpy()
    adaptive_pitch = df["adaptive_fused_pitch_deg"].to_numpy()
    kalman_pitch = df["kalman_angle_deg"].to_numpy()

    return {
        "gyro_pitch_rmse_deg": rmse_deg(gyro_pitch, gravity_pitch),
        "fixed_complementary_pitch_rmse_deg": rmse_deg(fixed_pitch, gravity_pitch),
        "adaptive_complementary_pitch_rmse_deg": rmse_deg(adaptive_pitch, gravity_pitch),
        "kalman_pitch_rmse_deg": rmse_deg(kalman_pitch, gravity_pitch),
        "kalman_bias_start_deg_s": float(df["kalman_bias_deg_s"].iloc[0]),
        "kalman_bias_end_deg_s": float(df["kalman_bias_deg_s"].iloc[-1]),
        "kalman_angle_start_deg": float(df["kalman_angle_deg"].iloc[0]),
        "kalman_angle_end_deg": float(df["kalman_angle_deg"].iloc[-1]),
    }

def save_kalman_pitch_plot(
    df: pd.DataFrame,
    output_dir: Path,
    experiment_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    time = df["time_sec"]

    plt.figure(figsize=(12, 5))
    plt.plot(time, df["pitch_deg"], label="gravity pitch")
    plt.plot(time, df["gyro_pitch_deg"], label="gyro-only pitch")
    plt.plot(time, df["fused_pitch_deg"], label="fixed complementary pitch")
    plt.plot(time, df["adaptive_fused_pitch_deg"], label="adaptive complementary pitch")
    plt.plot(time, df["kalman_angle_deg"], label="Kalman pitch")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch angle (deg)")
    plt.title(f"{experiment_id}: Pitch Angle Estimates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_id}_kalman_pitch_comparison.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(time, df["kalman_bias_deg_s"], label="estimated gyro bias")
    plt.xlabel("Time (s)")
    plt.ylabel("Bias estimate (deg/s)")
    plt.title(f"{experiment_id}: Kalman Estimated Gyro Bias")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_id}_kalman_bias.png")
    plt.close()



def main():
    df = pd.read_csv("output/IMU_tilt_test_data.csv")
    best_params = sweep_kalman_parameters(
        df,
        gyro_col="gyro_y_calibrated",
        measurement_col="pitch_rad",
        q_angle_values=[1e-6, 1e-5, 1e-4,1e-3],
        q_bias_values=[1e-8, 1e-7, 1e-6, 1e-5],
        r_values=[1e-4, 1e-3, 1e-2, 1e-1],
    )
    print(best_params.head(10).to_string(index=False))
    df = run_kalman_angle_bias(
        df,
        gyro_col="gyro_y_calibrated",
        measurement_col="pitch_rad",
        q_angle=best_params["q_angle"].iloc[0],
        q_bias=best_params["q_bias"].iloc[0],
        r_measurement=best_params["r_measurement"].iloc[0],
    )
    summary = summarize_kalman_performance(df)
    print(json.dumps(summary, indent=2))
    save_kalman_pitch_plot(df, output_dir=Path("output"), experiment_id="imu_tilt_test")


if __name__ == "__main__":
    main()