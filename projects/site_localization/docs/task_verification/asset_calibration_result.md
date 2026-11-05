# Conventional commissioning correction implemented; physical transfer unvalidated

The new component calibrates from printed graduations, registers subsequent images automatically, and reads in the commissioning coordinate frame. It requires no manual per-query center, zero, endpoints or correspondences. **It is not yet ready to serve as a verified measurement component in a robot/remapping experiment.** There are no eligible independent images establishing transfer to another observation of this physical asset. Do the small same-asset camera validation before navigation integration; no generic-reader expansion is justified.

## What changed and what stayed fixed

New implementation: [commissioned_gauge.py](../../research/task_verification/commissioned_gauge.py), with [commissioning](../../research/task_verification/commission_r02.py) and [bounded checks](../../research/task_verification/check_asset_calibration.py). The original reader is imported from a byte-identical frozen copy; detection thresholds, full angular search and original outputs remain unchanged. Only the returned angle-to-value mapping changes, after automatic reference-image registration. Out-of-range detections abstain rather than extrapolate.

The [pre-execution protocol](asset_calibration_protocol.md) and [frozen bundle](../../results/asset_calibration_20261023/frozen_bundle/manifest.json) record settings and source hashes. Manifest SHA256: `9baea88ebe09e0a733b92969c6dd836c2938ee30f2c96ac339f6f11e995be16c`. It includes executable source snapshots, reference pixels, graduation inputs and original detector. No tuning followed the three checks. All157 protected original files match their prior hashes, including the frozen reader,5.625MPa prediction, human[6,6.5]MPa interval and R02 diagnosis. The original result remains unresolved; this new development result does not replace it.

## Commissioning and runtime contract

| Item | Commissioning input / estimate |
|---|---|
| Asset association | R02 reference image; actual unit ID **unknown**, not inferred from appearance |
| Reference chart | Lossless425×425 crop at original offset(540,210) |
| Center/radius | Existing manually supplied center(214,210), radius165px; not re-estimated or optimized |
| Printed marks | Visible5MPa midpoint(54,219),10MPa midpoint(114,88.5) |
| Estimated conversion | Directions176.780505° and230.544073°; slope0.092999780MPa/degree |
| Operating range | Local5–10MPa only; no full-range claim |
| Needle estimator | Existing270° search retained; old zero only defines its search grid, not the new value calibration |
| Static registration | SIFT on the face annulus, ratio0.7/RANSAC3px,10matches/8inliers and inlier hull surrounding pivot |

The printed tick coordinates reuse the assistant's image annotations from the diagnosis; the user confirmed the scale and units, not surveyed pixel coordinates. Two major tick midpoints calibrate the mapping; nine interior ticks are excluded from fitting. Neither the5.625 output nor the human pressure interval enters commissioning or runtime. Calibration is `5 + (theta − 176.780505)*0.092999780`, with theta in reference image coordinates. No offset was fitted to a desired reading.

One-time assistance consists of selecting the asset/crop, placing a center and radius, marking two readable graduations, and confirming their values, units and intended range. These were reused here; no fresh operator session or measured commissioning time exists. Approximate prior placement uncertainty is±2px for marks and±3px for center. Those are sensitivity assumptions, not statistical bounds. No human input was supplied to the runtime checks.

Runtime inputs are **image + frozen asset bundle**. SIFT/RANSAC estimates H(reference→query); inverse warping returns query pixels to the fixed reference before the unchanged detector measures the needle. The runtime never receives the synthetic ground-truth transform. Numerical acceptance means a conditional reading only: outputs explicitly retain `asset_identity_verified=false` and `physical_completion_verified=false`.

## Geometry audit

Raw angles are not transferable between perspective views. In the controlled warp, a190° reference ray appears at186.943° even using the correctly transformed center. Inverse warping restores189.999995°. The chosen correction is the established planar image homography, not an assumed adjacent-marker plane. See OpenCV's [geometry explanation](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html) and [SIFT/RANSAC implementation example](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html).

This is reference-chart rectification, **not** recovery of a metric front-facing dial. The commissioning image's own perspective remains, so linear angular interpolation is local and empirically checked against visible graduations. A changed center changes all angles: for a fixed190° test ray,±3px center perturbations produce6.049–6.419MPa if the old mapping is reused, versus6.153–6.311 after recalibrating both anchors (nominal6.229). Even recalibration does not eliminate center error. Runtime therefore keeps the commissioned center in the rectified chart; replacing it requires recommissioning.

Required assumptions: planar rigid face, adequate static texture, negligible unmodeled lens distortion, and sufficiently small pointer-height/glass/parallax effects. None of the latter is measured here. The annular feature mask reduces moving-needle influence but is not a verified segmentation of static paint: tip/descriptor patches can still overlap moving features. RANSAC does not certify independence or identity. Similar dials could register convincingly. Real changed-pointer and viewpoint observations are needed; do not interpret the successful image warp as testing these assumptions.

## Measured development results

| Check | Result | Meaning |
|---|---|---|
| Nine unused intermediate graduations | Max absolute error0.01560MPa;7.5 mark maps7.49294 | Local scale consistency, not reading accuracy or new human truth |
| R02 self-registration |6.204152MPa;125/125 inliers;52ms | Correction works on its commissioning/development image |
| One synthetic planar warp |6.204152MPa;54/58 inliers;73ms | Software transfer works for this generated planar transform |
| Synthetic independent geometric checkpoints | Max0.2733 reference pixels | Checks transform direction and recovered chart, using evaluator-only known H |
| Blank negative | Abstain: no features;53ms | Basic missing-evidence behavior |

Nine mark checks use approximate observations from the same image and annotator. Their small residual is not a0.016MPa accuracy guarantee. Three attempted image executions, no retries or infrastructure failures; two conditional accepts and one abstention. The second positive is generated from the first, so these are **one real image, zero new real viewpoints, zero independent validation trials**. Timings exclude loading, one-time feature setup and operator work; CPU/OpenCV, no GPU.

The new R02 development value versus the unchanged human[6,6.5] interval has possible absolute error[0,0.29585]MPa, below the fixed0.5MPa tolerance for every value in that interval. This is a post-correction consistency comparison on the motivating image, not independent validation. There is no exact pressure reference, and neither a real-world incorrect-acceptance rate nor real-world abstention/coverage rate can be estimated. Do not call these three software outcomes deployments.

Raw evidence: [per-case outcomes](../../results/asset_calibration_20261023/checks/result.json), [every attempt](../../results/asset_calibration_20261023/checks/attempts.jsonl), [graduation checks](../../results/asset_calibration_20261023/checks/graduation_checks.json), [geometry/center sensitivity](../../results/asset_calibration_20261023/checks/geometry_checks.json), [preservation and bounds](../../results/asset_calibration_20261023/verification.json).

## Missing validation evidence and decision

The archive contains58 JPEGs and no asset/sequence/reference sidecars. EXIF is empty or only names an editing application; JPEG comments identify a camera vendor, not an asset. R02/R03 share design/blue surroundings; R08/R09 share a design with different surrounding views. No verified serial or capture lineage links any to R02. Similarities and filename dates are insufficient. [Full metadata audit](../../results/asset_calibration_20261023/provenance_audit.json). The [official PaddleX description](https://github.com/PaddlePaddle/PaddleX/blob/release/2.1/examples/meter_reader/README.md) explicitly describes these test images as lacking ground-truth labels. Existing human readings on other images cannot repair the missing asset association.

**The immediate bottleneck is evidence for calibration transfer and physical image geometry, not an observed needle-detection failure.** This does not establish that needle detection remains adequate under new conditions. Keep this small conventional component; pause algorithm changes until transfer can be judged.

The smallest useful next validation requires one traceably identified physical gauge with a commissioning image and separately captured test observations: one nominal repeat and one modest oblique/displaced view, with a recorded link proving they are the same unit, independently assigned reading intervals, and documented pressure stability or synchronized readings. Include one unusable-view negative if testing abstention. Reference intervals must be recorded without new-reader outputs; pressure values used for commissioning are printed ticks, not test readings. An actual change of pointer position would additionally test whether registration depends on the old pointer. Neither a robot nor an adjacent marker is necessary for this camera check. No hardware request, collection, download or outreach was made in this task.

Until such observations exist, **not ready for a robot experiment claiming verified measurement after remapping**. The code is ready for that bounded camera validation. It contributes a concrete geometry/calibration component and reproducible engineering evidence; no original research result or product demand is established. Do not integrate it into a physical task-completion gate based on these software checks alone.

## Reproduce from repository root

Inspect existing outputs first. Commands below use new output paths and refuse overwrite:

```bash
python3 projects/site_localization/results/asset_calibration_20261023/frozen_bundle/check_asset_calibration.py \
  --bundle projects/site_localization/results/asset_calibration_20261023/frozen_bundle \
  --out /tmp/r02_calibration_checks_repeat

python3 projects/site_localization/results/asset_calibration_20261023/frozen_bundle/commissioned_gauge.py \
  --bundle projects/site_localization/results/asset_calibration_20261023/frozen_bundle \
  --image projects/site_localization/results/asset_calibration_20261023/checks/synthetic_planar_warp.png \
  --out /tmp/r02_calibration_read_repeat.json
```

Environment used: Python3.12.3, OpenCV5.0.0, NumPy2.2.6, Pillow, existing Python environment; no install. The manifest checks calibration, source and image hashes at runtime. To rebuild a new commissioning bundle from the existing observations, run `research/task_verification/commission_r02.py --out <new-directory>` from the project directory (or use the full repository-relative script path from the repository root). The rebuild program never reads pressure-reference labels or predictions.
