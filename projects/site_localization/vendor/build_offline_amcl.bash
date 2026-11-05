#!/usr/bin/env bash
set -eo pipefail
vendor_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${vendor_dir}/setup_amcl.bash"
g++ -std=c++17 -O2 -Wall -Wextra -I"${AMCL_VENDOR_PREFIX}/include" "${vendor_dir}/offline_amcl.cpp" -L"${AMCL_VENDOR_PREFIX}/lib" -Wl,-rpath,"${AMCL_VENDOR_PREFIX}/lib" -lmotions_lib -lsensors_lib -lpf_lib -lmap_lib -o "${vendor_dir}/offline_amcl"
