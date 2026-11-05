# Known-asset calibration: bounded implementation protocol

Written before implementation/execution. Preserve the R02 diagnosis and all earlier reader, label and result files. This is a new conventional development component, not a new reading evaluation.

## Inputs, parameters and transfer

Commissioning supplies one reference image/crop, pivot, scale radius, two visible printed graduation stroke midpoints with values (5 and10 MPa), units and valid working interval [5,10]. Use the existing R02 image observations, not its pointer/reference reading, to derive the local slope and intercept. Those pixel locations are assistant visual annotations, not independently surveyed coordinates; the user confirmed the numbered scale and MPa units. Keep the original center and needle search settings. No fitting to6..6.5 or5.625. Intermediate marks, including7.5, are unused by calibration and check local interpolation consistency. They were already inspected in development and are not held-out data.

Runtime takes an image and the frozen commissioning bundle only: no per-query center, endpoints, zero or correspondences. SIFT descriptors on the static dial scale, ratio matching and RANSAC estimate H from reference pixels to current pixels. Inverse-warp the current image into the reference image before using the unchanged needle estimator. This transfers a fixed coordinate chart, not a raw current-image angle. Map the detected reference angle using the two commissioned marks; abstain outside the commissioned range. Keep the original full needle search, rather than searching only where a desired reading is expected.

H assumes the matched dial face is planar and rigid. No adjacent marker is used or assumed coplanar. It does not remove unknown distortion already in the reference, lens distortion, pointer height/parallax, glass refraction or face deformation. Local interpolation is an empirical approximation checked against unused printed marks. Changing/re-estimating the center requires recomputing the mark angles/calibration. A conic alone does not justify a metric rectification here; no invented camera calibration or3D plane is supplied.

Natural-feature registration does not prove asset identity among identical gauges. Runtime outputs are conditional readings, not physical task-completion certificates. A deployment still needs traceable asset association. Center/radius, two stroke locations and value/unit confirmation are one-time operator inputs; time was not measured. No later human geometry is allowed.

## Evidence eligibility

Inspected the58 JPEG members of the existing PaddleX test archive and their metadata, plus R02/R03/R08/R09 labeled crops. Archive has no identity/sequence/reference sidecars. Metadata does not establish identity. R02/R03 similar face/blue equipment, and similar R08/R09 markings, are insufficient. No verified unit serial/capture lineage found. Thus zero independent labeled observations of the R02 asset are currently eligible. Official dataset description explicitly says test images lack ground truth: https://github.com/PaddlePaddle/PaddleX/blob/release/2.1/examples/meter_reader/README.md . Do not run unrelated gauges or infer identity from image matches.

## Bounded development checks

Question: does graduation-based conversion repair the local mapping, and does reference-frame registration preserve it under an exact planar image warp without manual runtime geometry?

Predictions: if calibration caused the discrepancy, calibration on5/10 will also map unused interior graduations consistently. If viewpoint correction has the wrong direction/frame, a projectively warped copy will change the recovered reading or geometric checkpoints. If the center changes, recalculation from both marks and needle is needed; sensitivity will be reported, not tuned away. A blank image should abstain.

Freeze code, input crop, calibration, detector hash and geometric matching settings before running: one self-registration image, one specified planar warp of it, and one blank negative. The warp maps reference corners(0,0),(424,0),(424,424),(0,424) to(40,28),(405,8),(438,413),(15,442), output480x480. These are software checks, not independent observations or physical viewpoints. Exact warp is evaluator-only. No synthetic pressure/needle modification. Record all attempts; no tuning after outputs. Stops: after these three cases and geometric/center sensitivity checks, report failures rather than expanding the sweep.

Use established SIFT ratio0.7 and RANSAC3px; require10 matches/8 inliers as in the existing reference matcher, unique matched runtime features and inlier hull enclosing the reference pivot to reject one-sided/degenerate fits. These are engineering guards, not calibrated uncertainty guarantees. Needle thresholds stay unchanged. Runtime abstains without sufficient registration, framing, unique needle or in-range calibration. No threshold validation claim.

Predeclared software expectations: unused-mark error <=0.5MPa (unchanged2% of25MPa); invertible forward/inverse transform; recovered synthetic center/mark checkpoints within3 reference pixels; image-warp reading difference <=0.5MPa when both accept; blank abstains. Report narrower observed discrepancies without redefining the criterion. Audit center shifts of±3px (existing stated placement uncertainty) with and without recalibration; this is sensitivity, not a confidence interval.

Freeze before checking. No independent reading metrics unless same-asset eligibility is established; current count is zero. R02's existing interval can describe the new development output's error bounds, but cannot validate a correction motivated by this very image. Preserve the original5.625 prediction and unresolved classification.

Established geometry: OpenCV [feature matching and homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html) and [planar homography assumptions](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html). Our engineering adaptation is restricting matching to fixed face features and calibrating a local value interval. No originality claimed.
