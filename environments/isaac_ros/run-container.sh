#!/usr/bin/env bash
# Start (or attach to) the Isaac ROS container used by pipelines II and III.
#
# Equivalent to the original run_dev_zed.sh. The two bind mounts are required:
# the ZED SDK reads its per-camera calibration from settings/ and its neural
# resources from resources/, and neither is baked into the image. Without them
# the SVO2 playback fails in ways that look like a corrupt recording.
#
# Run ./apply-patches.sh and ./build-image.sh first.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-$(cd "$HERE/../.." && pwd)}"
IMAGE_KEY="${IMAGE_KEY:-ros2_humble.realsense.zed.custom}"

for d in /usr/local/zed/settings /usr/local/zed/resources; do
  [ -d "$d" ] || echo "warning: $d missing on the host; the ZED SDK will not find its $(basename "$d")" >&2
done

cd "$WS/src/isaac_ros_common"
exec ./scripts/run_dev.sh -b -i "$IMAGE_KEY" \
  -a "-v /usr/local/zed/settings:/usr/local/zed/settings \
      -v /usr/local/zed/resources:/usr/local/zed/resources"
