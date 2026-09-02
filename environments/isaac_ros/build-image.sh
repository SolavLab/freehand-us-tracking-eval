#!/usr/bin/env bash
# Build the container image. Building and starting are the same operation in
# Isaac ROS's run_dev.sh (-b builds then enters), so this delegates.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run-container.sh" "$@"
