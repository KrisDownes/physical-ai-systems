import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

dt = 0.1 #time step in seconds
duration = 20.0 
n = int(duration / dt)

t = np.arange(n) * dt

true_position = np.zeros(n)
true_velocity = np.zeros(n)


true_position[0] = 0.0
true_velocity[:] = 1.0
for k in range(1,n):
    true_position[k] = true_position[k-1] + true_velocity[k-1] * dt

#Noise
rng = np.random.default_rng(seed=0)
measurment_noise_std = 2.0
measurements = true_position + rng.standard_normal(n) * measurment_noise_std

#Kalman filter setup
#x_hat = [position, velocity]
x_hat = np.array([[0.0], [0.0]]) #Initial state estimate
F = np.array([[1, dt], [0, 1]]) #State transition matrix
H = np.array([[1, 0]]) #Measurement matrix
P = np.array([[10,0], [0,10]]) #Cov Matrix

sigma_a = 0.5
Q = sigma_a**2 * np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]]) #Process noise covariance
R = np.array([[measurment_noise_std**2]]) #Measurement noise covariance
I = np.eye(2)

estimated_position = np.zeros(n)
estimated_velocity = np.zeros(n)
position_uncertainty = np.zeros(n)

for k in range(n):
    x_pred = F @x_hat
    P_pred = F @ P @F.T + Q
    y = measurements[k] - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_hat = x_pred + K @ y
    P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T

    estimated_position[k] = x_hat[0,0]
    estimated_velocity[k] = x_hat[1,0]
    position_uncertainty[k] = np.sqrt(P[0,0])

#Save results
output_dir = Path("output/tiny_kalman_1d")
output_dir.mkdir(parents=True, exist_ok=True)

records_path = output_dir / "states.jsonl"
with open(records_path, "w") as f:
    for k in range(n):
        record = {
            "time_sec": float(t[k]),
            "true_position": float(true_position[k]),
            "measured_position": float(measurements[k]),
            "estimated_position": float(estimated_position[k]),
            "estimated_velocity": float(estimated_velocity[k]),
            "position_uncertainty": float(position_uncertainty[k]),
            "position_error": float(estimated_position[k] - true_position[k])
        }
        f.write(json.dumps(record) + "\n")



#Plot

plt.figure(figsize=(12, 6))

plt.plot(t, true_position, label="true position")
plt.scatter(t, measurements, s=10, label="noisy measurements")
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
plt.title("Tiny 1D Kalman Filter")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "tiny_kalman_1d.png")
plt.show()

rmse_measurement = np.sqrt(np.mean((measurements - true_position) ** 2))
rmse_estimate = np.sqrt(np.mean((estimated_position - true_position) ** 2))

print(f"Measurement RMSE: {rmse_measurement:.3f}")
print(f"Kalman estimate RMSE: {rmse_estimate:.3f}")
print(f"Saved records to {records_path}")