# Phone IMU Sensor Pipeline

## Goal

Build an end-to-end IMU data pipeline using phone sensors from phyphox.

The pipeline records accelerometer, gravity, and gyroscope data, saves raw sensor files, builds an aligned processed dataset, estimates orientation, removes gravity, classifies motion events, and validates the detected events against an expected experiment protocol.

This project is a robotics/autonomy learning project focused on:

* sensor ingestion
* time-series alignment
* calibration
* state estimation
* gravity removal
* event classification
* experiment validation
* reproducible data pipeline design

## Pipeline

```text
phyphox phone recording
→ raw accelerometer/gravity/gyroscope CSVs
→ aligned processed IMU dataset
→ gyro bias calibration
→ quaternion orientation estimate
→ world-frame acceleration
→ gravity removal
→ motion event classification
→ protocol validation
```

## Folder Layout

```text
app/
  fetch_phyphox_run.py
  build_processed_imu_dataset.py
  quaternion_experiment.py
  validate_motion_segments.py
  run_imu_pipeline.py

data/
  protocols/
  processed/

outputs/<run_id>/
  csv/
  plots/
  summaries/
  validation/
```

Raw phyphox files are saved outside the repo:

```text
/mnt/d/PhoneTelemetry/inbox/<run_id>/
```

Generated processed files and outputs should not be committed unless intentionally saving an example result.

## Setup

Start phyphox on the phone, enable remote access, and make sure the phone and computer are on the same network.

Test the connection:

```bash
export PHY_URL="http://192.168.50.75"

curl "$PHY_URL/config" | head
```

From the project directory:

```bash
cd ~/physical-ai-systems/projects/02-real-sensor-data-pipeline
```

## Example Protocol

This protocol uses wide timing windows because the phone is moved by hand.

```bash
RUN_ID="imu_validation_test_002"

mkdir -p data/protocols

cat > "data/protocols/$RUN_ID.csv" << 'EOF'
start_time_sec,end_time_sec,expected_label,notes
0,5,calibration_stationary,phone still for gyro bias calibration
5,8,transition,human timing buffer
8,15,rotation,slow rotation
15,22,motion,gentle hand motion
22,29,shake,active shaking motion
29,32,transition,stop moving and place phone still
32,40,stationary,phone still at end
EOF
```

Physical sequence:

```text
0–5s      phone still
5–8s      transition / get ready
8–15s     slow rotation
15–22s    gentle motion
22–29s    active shake
29–32s    transition / stop moving
32–40s    phone still
```

## Run Full Pipeline

```bash
python app/run_imu_pipeline.py \
  --url http://192.168.50.75 \
  --run-id "$RUN_ID" \
  --duration 40 \
  --raw-out-root /mnt/d/PhoneTelemetry/inbox \
  --bias-window-sec 5 \
  --protocol "data/protocols/$RUN_ID.csv"
```

## Expected Outputs

Raw files:

```text
/mnt/d/PhoneTelemetry/inbox/imu_validation_test_002/
  imu_validation_test_002_accelerometer.csv
  imu_validation_test_002_gravity.csv
  imu_validation_test_002_gyroscope.csv
  imu_validation_test_002_phyphox_buffers.json
```

Processed dataset:

```text
data/processed/imu_validation_test_002.csv
```

Analysis outputs:

```text
outputs/imu_validation_test_002/
  csv/
  plots/
  summaries/
  validation/
```

## Example Validation Result

The latest successful test run produced:

```text
Protocol validation:
 start_time_sec  end_time_sec         expected_label dominant_detected_label allowed_overlap_sec match_fraction  status
            0.0           5.0 calibration_stationary              stationary               4.979          0.996    pass
            5.0           8.0             transition                rotation               0.000                ignored
            8.0          15.0               rotation                rotation               6.198          0.885    pass
           15.0          22.0                 motion                rotation               6.980          0.997    pass
           22.0          29.0                  shake         impact_or_shake               6.960          0.994    pass
           29.0          32.0             transition                rotation               0.000                ignored
           32.0          40.0             stationary              stationary               5.703          0.713    pass
```

Summary:

```json
{
  "protocol_rows": 7,
  "evaluated_rows": 5,
  "ignored_rows": 2,
  "passed_rows": 5,
  "failed_rows": 0,
  "pass_rate": 1.0
}
```

This confirms that the pipeline can collect live phone IMU data, process it, classify broad motion phases, and validate the output against a realistic hand-run experiment protocol.

## Core Math

Gyro calibration:

```text
gyro_calibrated = gyro_measured - gyro_bias
```

Gyro integration:

```text
angle[k] = angle[k-1] + gyro_calibrated[k] * dt[k]
```

Gravity-derived roll and pitch:

```text
roll = atan2(gravity_y, gravity_z)

pitch = atan2(-gravity_x, sqrt(gravity_y² + gravity_z²))
```

Complementary filter:

```text
fused[k] = alpha * (fused[k-1] + gyro_calibrated[k] * dt[k])
           + (1 - alpha) * gravity_angle[k]
```

Quaternion pipeline:

```text
gyro angular velocity
→ delta quaternion
→ orientation quaternion
→ rotation matrix
→ world-frame acceleration
→ gravity-removed linear acceleration
```

## Earlier Attitude Estimation Result

| Method                        | Roll RMSE | Pitch RMSE |
| ----------------------------- | --------: | ---------: |
| Gyro-only integration         |     9.71° |      2.51° |
| Fixed complementary filter    |     1.50° |      1.41° |
| Adaptive complementary filter |     0.96° |      0.74° |

Best adaptive parameters:

```text
alpha_normal = 0.90
alpha_high_accel = 0.995
acc_threshold = 5.0 m/s²
```

## Interpretation

Gyro-only integration drifts over time. Gravity-derived roll and pitch provide a stable correction reference. The complementary filter reduces drift by blending gyro prediction with gravity correction. The quaternion pipeline extends this into 3D orientation estimation, world-frame acceleration, gravity removal, and motion event classification.

The final validation layer checks whether the detected motion phases match the intended protocol, making the pipeline testable instead of just runnable.
