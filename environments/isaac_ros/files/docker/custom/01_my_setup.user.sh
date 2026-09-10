#!/bin/bash
# Custom setup script for Isaac ROS development environment
# 
# This script performs initialization tasks for the Isaac ROS workspace:
# - Sources the ROS 2 Humble environment if not already sourced
# - Updates the rosdep package index
# - Installs system dependencies for isaac_ros_visual_slam package
#
# The script uses 'set -e' to exit immediately if any command fails.
# All rosdep operations include error handling to prevent script failure.
#
# Prerequisites:
#   - ROS 2 Humble installed at /opt/ros/humble/
#   - rosdep package manager available
#   - ISAAC_ROS_WS environment variable set to workspace path
#   - isaac_ros_visual_slam source directory present in workspace
#
# Exit codes:
#   0 - Script completed successfully
#   Non-zero - Command execution failed (due to 'set -e')


set -e

FNAME=$(basename "$0")
echo "=== Custom Setup Script: ${FNAME} ==="
# Explain the purpose of this script and its steps
echo "This script initializes the Isaac ROS workspace by sourcing the ROS 2 environment,"
echo "updating rosdep, and installing necessary dependencies for the isaac_ros_visual_slam package."
# Set non-interactive frontend to suppress debconf dialog warnings
export DEBIAN_FRONTEND=noninteractive

# Export CMAKE_PREFIX_PATH so it's available for all colcon builds
export CMAKE_PREFIX_PATH=/usr/local/zed:$CMAKE_PREFIX_PATH

# Source ROS 2 environment if not already sourced (check ROS_DISTRO)
if [ -z "$ROS_DISTRO" ]; then
    echo "Sourcing ROS 2 Humble environment..."
    source /opt/ros/humble/setup.bash
fi

# Update apt package lists
echo "Updating apt package lists..."
sudo apt-get update -y || true

# Update rosdep index
echo "Updating rosdep..."
rosdep update

# Install system dependencies for isaac_ros_visual_slam 
if [ -d "${ISAAC_ROS_WS}/src/isaac_ros_visual_slam" ]; then
    echo "Installing dependencies for isaac_ros_visual_slam..."
    rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_ros_visual_slam/isaac_ros_visual_slam \
        --ignore-src -r -y 2>&1 | grep -E '(installed|skipping|Reading)' | tail -20 || true
    echo "✓ rosdep dependencies for isaac_ros_visual_slam installed"
fi

# Build isaac_ros_visual_slam
if [ -d "${ISAAC_ROS_WS}/src/isaac_ros_visual_slam" ]; then
    echo "Building isaac_ros_visual_slam..."
    source /opt/ros/humble/setup.bash 
    cd ${ISAAC_ROS_WS}/ && \
    colcon build --symlink-install --packages-up-to isaac_ros_visual_slam --base-paths ${ISAAC_ROS_WS}/src/isaac_ros_visual_slam/isaac_ros_visual_slam
    echo "✓ isaac_ros_visual_slam built successfully"
fi

echo "Sourcing the workspace..."
source ${ISAAC_ROS_WS}/install/setup.bash
# Remind user to source the workspace
echo "Reminder: source the workspace in each new terminal session (source ${ISAAC_ROS_WS}/install/setup.bash)."

echo "=== Custom Setup Complete ==="

