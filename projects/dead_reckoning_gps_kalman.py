import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Sim Setup

dt = 0.1
duration = 60
n = int(duration / dt)

t = np.arange(n) * dt
rng = np.random.default_rng(seed=0)

#True state
x_true = np.zeros(n)
v_true = np.zeros(n)

# Simulate changing true velocity
v_true = np.select([t < 15, t < 30, t < 45],[1.0, 0.4, 1.5], default=0.8)
x_true[1:] = np.cumsum(v_true[:-1] * dt)

# Simulate sensor measurements
# Velocity sensor / odometry
# Bias is important: even a small bias will cause large position errors over time

v_noise_std = 0.15
v_bias = 0.05
v_measurements = (v_true + v_bias + v_noise_std * rng.standard_normal(n))

# GPS position measurements
gps_noise_std = 2.0
gps_period_sec = 1.0
gps_period_steps = int(gps_period_sec / dt)

gps_measurements = np.full(n, np.nan)
for k in range(0, n, gps_period_steps):
    gps_measurements[k] = x_true[k] + gps_noise_std * rng.standard_normal()

#Dead reckoning from Velocity measurements
dead_reckoning_position = np.zeros(n)
for k in range(1,n):
    dead_reckoning_position[k] = (dead_reckoning_position[k-1] + v_measurements[k-1] * dt)

# Kalman Filter Setup

x_hat = np.array([0.0, 0.0])
F = np.array([[1.0, dt], [0.0, 1.0]])
P = np.array([[10.0, 0.0], [0.0, 10.0]])
Q = np.array([[0.02, 0.0], [0.0, 0.05]])
I = np.eye(2)

estimated_position = np.zeros(n)
estimated_velocity = np.zeros(n)
position_uncertainty = np.zeros(n)

for k in range(n):
    #Predict using motion model
    x_pred = F @ x_hat
    P_pred = F @ P @ F.T + Q
    #Update using velocity measurement
    H_vel = np.array([[0.0, 1.0]])
    R_vel = np.array([[v_noise_std**2]])
    z_vel = np.array([v_measurements[k]])
    y = z_vel - H_vel @ x_pred
    S = H_vel @ P_pred @ H_vel.T + R_vel
    K = P_pred @ H_vel.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    P_upd = (I - K @ H_vel) @ P_pred
    #Update using GPS measurement if available
    if not np.isnan(gps_measurements[k]):
        H_gps = np.array([[1.0, 0.0]])
        R_gps = np.array([[gps_noise_std**2]])
        z_gps = np.array([gps_measurements[k]])
        y_gps = z_gps - H_gps @ x_upd
        S_gps = H_gps @ P_upd @ H_gps.T + R_gps
        K_gps = P_upd @ H_gps.T @ np.linalg.inv(S_gps)
        x_upd += K_gps @ y_gps
        P_upd = (I - K_gps @ H_gps) @ P_upd
    #Save estimates and uncertainties
    x_hat = x_upd
    P = P_upd
    estimated_position[k] = x_hat[0]
    estimated_velocity[k] = x_hat[1]
    position_uncertainty[k] = np.sqrt(P[0,0])

#Metrics
gps_available = ~np.isnan(gps_measurements)
dead_reckoning_rmse = np.sqrt(np.mean((dead_reckoning_position - x_true)**2))
kalman_rmse = np.sqrt(np.mean((estimated_position - x_true)**2))
velocity_rmse = np.sqrt(np.mean((v_measurements - v_true)**2))
fused_velocity_rmse = np.sqrt(np.mean((estimated_velocity - v_true)**2))
gps_rmse = np.sqrt(np.mean((gps_measurements[gps_available] - x_true[gps_available])**2))

output_dir = Path("outputs/dead_reckoning_gps_kalman/run_001")
output_dir.mkdir(parents=True, exist_ok=True)

config = {
    "dt": dt,
    "duration": duration,
    "velocity_noise_std": v_noise_std,
    "velocity_bias": v_bias,
    "gps_noise_std": gps_noise_std,
    "gps_period_sec": gps_period_sec,
    "Q": Q.tolist(),
}
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f, indent=4)
metrics = {
    "dead_reckoning_position_rmse": float(dead_reckoning_rmse),
    "gps_position_rmse": float(gps_rmse),
    "kalman_position_rmse": float(kalman_rmse),
    "velocity_sensor_rmse": float(velocity_rmse),
    "fused_velocity_rmse": float(fused_velocity_rmse),
}
with open(output_dir / "states.jsonl", "w") as f:
    for k in range(n):
        record = {
            "time_sec": float(t[k]),
            "true_position": float(x_true[k]),
            "true_velocity": float(v_true[k]),
            "velocity_measurement": float(v_measurements[k]),
            "gps_measurement": (
                None
                if np.isnan(gps_measurements[k])
                else float(gps_measurements[k])
            ),
            "dead_reckoned_position": float(dead_reckoning_position[k]),
            "estimated_position": float(estimated_position[k]),
            "estimated_velocity": float(estimated_velocity[k]),
            "position_uncertainty": float(position_uncertainty[k]),
            "position_error": float(estimated_position[k] - x_true[k]),
        }

        f.write(json.dumps(record) + "\n")
#Plots
plt.figure(figsize=(12, 6))

plt.plot(t, x_true, label="true position")
plt.plot(t, dead_reckoning_position, label="dead reckoning")
plt.scatter(
    t[gps_available],
    gps_measurements[gps_available],
    s=15,
    label="GPS measurements",
)
plt.plot(t, estimated_position, label="Kalman estimate")

plt.fill_between(
    t,
    estimated_position - 2 * position_uncertainty,
    estimated_position + 2 * position_uncertainty,
    alpha=0.2,
    label="±2σ uncertainty",
)

plt.xlabel("Time (sec)")
plt.ylabel("Position")
plt.title("Dead Reckoning + GPS-Corrected Kalman Filter")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "position.png")
plt.show()


# -----------------------------
# 9. Plot velocity
# -----------------------------

plt.figure(figsize=(12, 6))

plt.plot(t, v_true, label="true velocity")
plt.scatter(t, v_measurements, s=5, label="velocity measurements")
plt.plot(t, estimated_velocity, label="fused velocity estimate")

plt.xlabel("Time (sec)")
plt.ylabel("Velocity")
plt.title("Velocity Estimate")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "velocity.png")
plt.show()


# -----------------------------
# 10. Print summary
# -----------------------------

print("Dead Reckoning + GPS Kalman Filter")
print("----------------------------------")
print(f"Dead reckoning RMSE: {dead_reckoning_rmse:.3f}")
print(f"GPS RMSE: {gps_rmse:.3f}")
print(f"Kalman position RMSE: {kalman_rmse:.3f}")
print(f"Velocity sensor RMSE: {velocity_rmse:.3f}")
print(f"Fused velocity RMSE: {fused_velocity_rmse:.3f}")
print(f"Saved outputs to {output_dir}")