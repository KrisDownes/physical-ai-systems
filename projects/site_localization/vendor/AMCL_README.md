# Official Nav2 AMCL offline adapter

This directory contains official Ubuntu Noble amd64 Nav2 Jazzy 1.3.13 binary packages, extracted without root installation. `amcl_packages.json` records exact apt versions, official URLs, file sizes and verified SHA256 values. Cached local apt metadata supplied the expected hashes. HTTPS to packages.ros.org returned a certificate hostname mismatch; downloads therefore used the repository's configured HTTP endpoint and verified its expected SHA256. This is hash verification against existing local apt metadata, not a fresh independent repository signature audit.

Download size: 2,404,470 bytes for four packages. System ROS Jazzy supplies the remaining dependencies. No global packages were changed. Official package license files remain under `amcl_root/opt/ros/jazzy/share/*/`; upstream AMCL core is LGPL-2.1-or-later.

```bash
python3 projects/site_localization/vendor/fetch_amcl_packages.py
source projects/site_localization/vendor/setup_amcl.bash
bash projects/site_localization/vendor/build_offline_amcl.bash
projects/site_localization/vendor/offline_amcl --help
projects/site_localization/vendor/offline_amcl MAP.txt FRAMES.txt OUT.csv INITIAL_X INITIAL_Y INITIAL_YAW 42 likelihood_field 1000 10000 0.2 0.2 100
```

Arguments after initial pose: seed, sensor model (`likelihood_field`, diagnostic `likelihood_field_prob`, or `likelihood_field_prob_noskip`), minimum and maximum particle counts, common motion alpha1..alpha5, sigma_hit, maximum beams, initial xy standard deviation (.2m default), initial yaw standard deviation (10degrees default). These initialization defaults follow the paper tracking condition rather than ROS node defaults. Common defaults: initial pose covariance diagonal=(0.04m²,0.04m²,(pi/18)²rad²), z_hit=z_rand=0.5, max obstacle distance2m, pf_err=.05, pf_z=.99, recovery alpha_slow=alpha_fast=0, resample every supplied scan. Probabilistic sensor uses beam_skip_distance=.5m, threshold=.3, error_threshold=.9. Parameters are explicit experimental choices, not tuned results. Both `pf_alloc` and stock Gaussian allocation reset the RNG. The adapter allocates the Gaussian PDF, then seeds, then calls stock `pf_init_model` with the stock Gaussian sampler. Tests verify that different seeds change output. Earlier adapter revisions failed this check; see `docs/baseline_review.md`. Beam-skipping fallback also has a confirmed upstream semantic defect; that configuration is excluded from primary results.

## Map input

Plain whitespace text:

```
WIDTH HEIGHT RESOLUTION ORIGIN_CORNER_X ORIGIN_CORNER_Y
CELL_0 CELL_1 ... CELL_(WIDTH*HEIGHT-1)
```

Map is row-major (`index=x+y*WIDTH`), y increases upward. States are AMCL's -1=free,0=unknown,+1=occupied, **not** ROS OccupancyGrid numeric codes. Corner origin is ROS map origin, map yaw must be zero. Adapter converts map center exactly as Nav2 `AmclNode::convertMap`: corner+(INTEGER width/2,height/2)*resolution. Images normally need vertical flip. Do not add half-cell offsets silently.

## Frames input

```
LASER_TRANSLATION_X LASER_TRANSLATION_Y LASER_YAW RANGE_MAX
TIMESTAMP ODOM_X ODOM_Y ODOM_YAW N RANGE_0 BEARING_0 ... RANGE_(N-1) BEARING_(N-1)
...
```

All SI units/radians; timestamp strictly increasing. Odometry is robot base pose in a fixed odom frame, sampled/interpolated at scan timestamp. Ranges and bearings are in the laser frame. The complete planar laser extrinsic maps laser frame into robot base; unlike the ROS node's internally pretransformed bearing representation, this adapter applies the extrinsic yaw in `SetLaserPose`. Supply every beam in order and let stock max_beams subsample. Only finite nonnegative values accepted. Input preparation must replace invalid/too-near ranges with RANGE_MAX and document clipping. RANGE_MAX and larger values are ignored by likelihood-field measurement model as in upstream. First frame performs only measurement; later frames supply odometry delta in odom coordinates to stock OmniMotionModel.

## Output and scope

CSV: timestamp,x,y,yaw,n_particles,cluster_weight,cov_xx,cov_yy,cov_yawyaw. Pose and covariance are the dominant particle cluster, matching stock AMCL's cluster selection. Cluster weight/covariance are filter diagnostics, not calibrated certificates of true error.

Core algorithms are the unmodified official `libmotions_lib.so`, `libsensors_lib.so`, `libpf_lib.so`, and `libmap_lib.so`. This adapter supplies file IO, initial distribution, deterministic seeding and update scheduling. It is **not the ROS lifecycle node and does not establish exact equivalence to a ROS replay**: TF lookup, minimum-motion thresholds, message filtering, scan preprocessing and lifecycle scheduling are absent. Every provided row causes sensor update and resampling. Use fairly chosen scan/update intervals; do not claim full Nav2 pipeline validation from this adapter.

Verified: all dynamic libraries resolve after sourcing setup; compiled with GCC using -O2/-Wall/-Wextra; --help operational; a 20-scan generated square-room smoke fixture produced finite outputs and identical CSV bytes across repeated seed42 runs. This smoke check verifies the link, parser and determinism, not localization effectiveness. Real recorded-data outcomes must be measured separately.

The official ROS executable also resolves its dependencies and reaches its inactive lifecycle state. It ignores --help as a standalone help request and attempts DDS initialization. Restricted networking reports socket errors, so that node is not used for the offline baseline. ROS default log directory is read-only in sandbox; any future ROS run should set ROS_LOG_DIR to a writable project or /tmp path.

Upstream source checked for adapter conventions:
- https://github.com/ros-navigation/navigation2/tree/1.3.13/nav2_amcl
- https://raw.githubusercontent.com/ros-navigation/navigation2/1.3.13/nav2_amcl/src/amcl_node.cpp
- https://raw.githubusercontent.com/ros-navigation/navigation2/1.3.13/nav2_amcl/src/pf/pf.c
- https://raw.githubusercontent.com/ros-navigation/navigation2/1.3.13/nav2_amcl/src/sensors/laser/likelihood_field_model.cpp
