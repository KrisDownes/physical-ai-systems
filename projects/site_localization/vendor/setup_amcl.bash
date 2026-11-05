# Source this script from bash. Uses extracted official ROS packages; no system installation.
source /opt/ros/jazzy/setup.bash
_amcl_vendor="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export AMCL_VENDOR_PREFIX="${_amcl_vendor}/amcl_root/opt/ros/jazzy"
export AMENT_PREFIX_PATH="${AMCL_VENDOR_PREFIX}:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${AMCL_VENDOR_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${AMCL_VENDOR_PREFIX}/lib/python3.12/site-packages:${PYTHONPATH:-}"
unset _amcl_vendor
