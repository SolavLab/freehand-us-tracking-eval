#!/usr/bin/env bash
# Apply this project's modifications to the pinned upstream Isaac ROS sources.
#
# Run after `vcs import ../../src < repos.yaml`. Idempotent: re-running on an
# already-patched tree reports that and changes nothing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-$(cd "$HERE/../.." && pwd)}"
COMMON="$WS/src/isaac_ros_common"

[ -d "$COMMON" ] || { echo "error: $COMMON not found -- run 'vcs import' first" >&2; exit 1; }

echo "workspace: $WS"
echo "target:    $COMMON ($(git -C "$COMMON" describe --tags --always 2>/dev/null || echo 'unknown revision'))"

for p in "$HERE"/patches/*.patch; do
  name="$(basename "$p")"
  if git -C "$COMMON" apply --check --reverse "$p" 2>/dev/null; then
    echo "  = $name already applied"
  elif git -C "$COMMON" apply --check "$p" 2>/dev/null; then
    git -C "$COMMON" apply "$p"
    echo "  + $name applied"
  else
    echo "  ! $name does not apply cleanly -- upstream has probably moved" >&2
    echo "    expected revision is pinned in repos.yaml" >&2
    exit 1
  fi
done

# New files this project adds. Copied rather than patched, since a patch that
# creates files is noisier to review than the files themselves.
while IFS= read -r -d '' f; do
  rel="${f#"$HERE/files/"}"
  mkdir -p "$COMMON/$(dirname "$rel")"
  cp "$f" "$COMMON/$rel"
  echo "  > $rel"
done < <(find "$HERE/files" -type f -print0)

# The image key selects which Dockerfile layers are composed, in order.
printf 'CONFIG_IMAGE_KEY=ros2_humble.realsense.zed.custom\n' \
  > "$COMMON/scripts/.isaac_ros_common-config"
echo "  > scripts/.isaac_ros_common-config (image key)"

echo "done. Next: ./build-image.sh"
