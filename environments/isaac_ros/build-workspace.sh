#!/usr/bin/env bash
# Build the ROS 2 workspace INSIDE the running container.
#
# --allow-overriding is needed because the Isaac ROS apt packages and the
# workspace sources provide some of the same package names; without it colcon
# refuses to build.
set -euo pipefail

CONTAINER="${CONTAINER_NAME:-isaac_ros_dev-x86_64-container}"
WS="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
PACKAGES="${*:-}"

docker exec -i --user admin "$CONTAINER" bash -lc "
  set -eo pipefail
  cd '$WS'
  source /opt/ros/humble/setup.bash
  colcon build --symlink-install \
    --allow-overriding isaac_ros_common isaac_ros_tensor_list_interfaces isaac_ros_test \
    ${PACKAGES:+--packages-select $PACKAGES}
  echo
  echo 'built. Source it with:  source $WS/install/setup.bash'
"
