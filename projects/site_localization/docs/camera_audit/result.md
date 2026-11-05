# Camera observation path audit

## Exact path

The image received by the detector is **not rendered by a Gazebo RGB camera**.
Gazebo simulates the rover, lidar, contacts and physical pose. The
[camera node](../../research/moved_asset/camera.py) subscribes to the
evaluator's `/judge/truth` Odometry and reads the private
`scene.json` containing exact asset poses, texture paths and box obstacles.
On `/site/camera_request` it waits for a truth pose at or after the request
time, calls [custom `render()`](../../research/moved_asset/render.py), and
saves an 8-bit grayscale lossless PNG to `inspection/camera/observation_NNN.png`.
These are requested still captures after navigation events, not a continuous
camera stream. The renderer places the optical center at the rover's planar
truth position with fixed height and yaw; it models no camera-to-body offset,
pitch/roll or extrinsic uncertainty.
It publishes only `{id, time, file}` on `/site/camera_result`; `id` is the
capture sequence number, **not an asset identity**. Truth-derived labels such
as `actual_id`, true asset pose and image visibility go only to
`inspection/private/sensor_truth.jsonl` for analysis. An [actual mission
launch record](../../results/fresh_inspection_aware_20261028/mission_wrong_asset_aware/stages.jsonl)
shows separate simulator, camera and inspector processes.

The [inspector](../../research/continuous_asset/inspection.py) resolves the
filename under its camera directory, invokes the unchanged
[pixel detector](../../research/autonomous_asset/vision.py) on that PNG, and
associates its timestamp with interpolated AMCL pose (≤0.15 s pose bracket;
≤5 s processing age). The detector decodes AprilTag 36h11 from pixels,
estimates planar pose using the commissioned marker size and ideal camera
matrix, and optionally passes a pixel ROI to the frozen gauge reader. The
controller subscribes to clock, wheel odometry, lidar and localization; it has
no `/judge/truth` subscription. The new belief selector reads the acquired
map, estimated pose and detector output; it has no scene/truth input.

The private scene and Gazebo SDF are prepared from the same scenario choices,
but are separate descriptions. The SDF asset panels have collision geometry
and **no visual mesh**; the camera does not read Gazebo's world visuals. A
future change made only to the SDF would not necessarily change camera pixels.
The current separation is enforced by audited code paths, not by OS file
permissions: private files reside in the same output tree, so a future policy
change could violate it. [Example old-view PNG with ID8](../../results/fresh_inspection_aware_20261028/mission_wrong_asset_aware/inspection/camera/observation_000.png)
→ [pixel detector output](../../results/fresh_inspection_aware_20261028/mission_wrong_asset_aware/inspection/vision_000.json)
→ [private evaluator metadata](../../results/fresh_inspection_aware_20261028/mission_wrong_asset_aware/inspection/private/sensor_truth.jsonl).

## What the image model represents

| Effect | Actual implementation |
|---|---|
| Perspective | Ideal pinhole projection, 60° horizontal FOV, 1920×1080 pixels, fixed 0.8 m camera height; planar asset texture warped by homography. |
| Occlusion | Per-pixel rays against specified 3-D axis-aligned box obstacles and a depth buffer among asset panels. The obstacle list is a separate private scene model, not the Gazebo depth image. |
| Marker and gauge | Exact generated 16 cm AprilTags ID7/ID8 and the same frozen 425×425 gauge reference pasted into flat 1000×600 textures at fixed physical scale and offset. Asset identity is encoded in marker pixels. Gauge needle/readout is static. |
| Resolution | 8-bit grayscale output, 1920×1080; perspective resampling and downsampling can make distant markings small. PNG storage is lossless. |
| Illumination | Fixed grayscale background/panel values and any appearance already baked into the reference texture. No scene lights, shadows, exposure or changing illumination. |
| Distortion | None: centered ideal pinhole intrinsics, zero distortion in pose solving. No rolling shutter. |
| Blur and noise | No modeled focus or motion blur, sensor noise, compression noise or temporal variation. Interpolation from warping is present. |

The [texture construction](../../research/moved_asset/study.py) uses the same
frozen gauge image for both assets and changes only marker ID. Thus the
synthetic reading is especially favorable to the commissioned template
reader. The independent physical judge is still valuable for pose, contact,
dwell and false-claim checks, but these missions do **not** validate a real
camera's marker detection, pose calibration or gauge-reading accuracy.

## Existing frame and detector commands

Frames are already saved automatically on each request. To inspect the
retained wrong-asset run:

```bash
ls projects/site_localization/results/fresh_inspection_aware_20261028/mission_wrong_asset_aware/inspection/camera/observation_*.png
```

To generate a new **simulated** set, from repository root, while no other
site-localization simulator is active:

```bash
mkdir -p projects/site_localization/results/camera_audit_repro
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 projects/site_localization/research/inspection_aware/run_continuous.py \
  --template projects/site_localization/results/fresh_inspection_aware_20261028/templates/wrong_asset_aware \
  --out projects/site_localization/results/camera_audit_repro/mission_01 \
  --slam-params projects/site_localization/research/constraint_repair/slam_params_radius_1p5.yaml \
  --inspector-script projects/site_localization/research/inspection_aware/inspection.py \
  --seed 7601 --wall-limit 800
ls projects/site_localization/results/camera_audit_repro/mission_01/inspection/camera/observation_*.png
```

The unchanged detector already accepts a still image. Use its configured
Python environment, and choose a new JSON output path:

```bash
projects/robot_policy_eval/.venv/bin/python3 \
  projects/site_localization/research/autonomous_asset/vision.py \
  --image /absolute/path/to/frame.png \
  --config projects/site_localization/results/fresh_inspection_aware_20261028/templates/visible_aware/public/vision.json \
  --out /tmp/site_external_still_prediction.json
```

The detector CLI assumes, but does not check, 1920×1080 pixels and matching
60° centered, undistorted camera intrinsics. For **external** stills or video,
use the new [minimal replay adapter](../../research/camera_audit/replay_external.py),
which verifies the frozen detector hash, refuses a wrong resolution, saves
selected lossless decoded frames, records input/config/detector hashes, and calls the unchanged detector. It
does **not** calibrate a real camera or supply reference labels:

```bash
projects/robot_policy_eval/.venv/bin/python3 \
  projects/site_localization/research/camera_audit/replay_external.py \
  --image /absolute/path/to/1920x1080-frame.png \
  --config projects/site_localization/results/fresh_inspection_aware_20261028/templates/visible_aware/public/vision.json \
  --out /tmp/site_external_image_replay_01

projects/robot_policy_eval/.venv/bin/python3 \
  projects/site_localization/research/camera_audit/replay_external.py \
  --video /absolute/path/to/recording.mp4 --every 30 --max-frames 20 \
  --config projects/site_localization/results/fresh_inspection_aware_20261028/templates/visible_aware/public/vision.json \
  --out /tmp/site_external_video_replay_01
```

Each output has `manifest.json` and `frames/frame_NNNNNN.png/.json`. `--inspect`
runs the same frozen commissioned reader, but a real reading is interpretable
only after independent commissioning establishes that the recording shows
the physical gauge, attached ID7 and offset represented by this bundle;
that same-asset condition has not been established. A correct
pixel size alone does not establish matching focal length, distortion,
camera-to-robot calibration or marker size. ID detections may be inspected
without those for a limited pixel-level test; meter-valued pose and a gauge
reading need the matching calibration/asset and independent labels. Video
frames from one recording are correlated observations, not independent trials.

## Interface check and next step

The adapter was checked on retained **synthetic** frames only. A still-image
adapter result exactly matched the frozen detector CLI on an ID8 image. A
three-frame lossless synthetic video produced ID7 detections, with two gauge
abstentions and one accepted final view. A 960×540 image was rejected before
output because it violates the frozen intrinsics. The unchanged detector
hash still matches the previous pipeline freeze. These are interface checks,
**not real-recording replay or accuracy evidence**.

No arbitrary blur/noise constant is justified by current observations, and a
Gazebo camera swap would remain synthetic while changing multiple subsystems.
The smallest useful next step is a **separate real-recording component replay**
with known camera intrinsics and independent asset/reading references, if
such recordings already exist. It would test detector and reader transfer
without claiming navigation success. No new robot or hardware is required for
the interface, and none is requested by this audit. The existing simulated
closed-loop mission results remain a separate evidence category.
