#!/bin/sh
set -eu
revision=370c43965323c0944b433b202aac5058b5f907a9
project=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
destination="$project/.so101-model"
if [ -e "$destination" ]; then
  echo "Model directory already exists: $destination" >&2
  exit 1
fi
tempdir=$(mktemp -d)
trap 'rm -rf "$tempdir"' EXIT
git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$tempdir/repo"
git -C "$tempdir/repo" sparse-checkout set robotstudio_so101
git -C "$tempdir/repo" checkout "$revision"
cp -a "$tempdir/repo/robotstudio_so101" "$destination"
echo "Model $revision installed at $destination"
