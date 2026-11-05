# Checkpoint contents

This checkpoint includes the simulation and perception source, frozen gauge
bundle, scenario templates, result summaries, per-mission outcomes, and
selected method and camera traces. The source gauge reference was derived
from the PaddleX `meter_test` data described in the calibration report.

The full local results archive includes large trajectory, mapper, and sensor
logs that are not in this commit. Re-running the historical analysis scripts
requires that archive. The recorded result summaries can be inspected without
it.

The frozen templates contain machine-specific paths, including the Python
environment and gauge bundle location. A new machine must update those paths
and build the local AMCL adapter before running a mission. The historical
freeze manifests describe the original environment and may not verify after
those changes.
