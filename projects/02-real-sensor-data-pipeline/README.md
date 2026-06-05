# Experiment 002: Phone IMU Attitude Estimation

## Goal
Estimate phone roll/pitch orientation from Phyphox IMU data and reduce gyro drift using gravity correction

## Sensors Used
Gyroscope, acceleration, gravity sensors from Phyphox simple experiment

## Core Math
gyro_calibrated = gyro_measured - gyro_bias

angle[k] = angle[k-1] + gyro_calibrated[k] * dt[k]

roll = atan2(gravity_y, gravity_z)

pitch = atan2(-gravity_x, sqrt(gravity_y² + gravity_z²))

fused[k] = alpha * (fused[k-1] + gyro_calibrated[k] * dt[k]) + (1 - alpha) * gravity_angle[k]

## Results

| Method | Roll RMSE | Pitch RMSE |
|---|---:|---:|
| Gyro-only integration | 9.71° | 2.51° |
| Fixed complementary filter | 1.50° | 1.41° |
| Adaptive complementary filter | 0.96° | 0.74° |

Best adaptive parameters:
- alpha_normal = 0.90
- alpha_high_accel = 0.995
- acc_threshold = 5.0 m/s²

## Interpretation
Gyro-only integration drifted over time. The fixed complementary filter reduced drift by correcting toward the gravity-derived angle. The adaptive filter improved further by trusting gravity most of the time and switching to gyro-dominant prediction only during extreme acceleration spikes.
