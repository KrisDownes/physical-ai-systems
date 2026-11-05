#!/usr/bin/env bash
set -euo pipefail
out=${1:?give output executable path}
includes=()
for dir in /opt/ros/jazzy/include/*; do
  if test -d "$dir"; then includes+=("-I$dir"); fi
done
g++ -std=c++17 -O2 -I/usr/include/eigen3 -I/opt/ros/jazzy/include \
  "${includes[@]}" \
  projects/site_localization/research/radius_limit/replay_events.cpp \
  -L/opt/ros/jazzy/lib -Wl,-rpath,/opt/ros/jazzy/lib \
  -lkartoSlamToolbox -lboost_serialization -ltbb -o "$out"
