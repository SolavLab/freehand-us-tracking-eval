#!/bin/bash

# Enable ZED if installed
echo "Checking ZED SDK..."

ZED_FOLDER=/usr/local/zed/

if [ -d "${ZED_FOLDER}" ]; then
   echo " * ${ZED_FOLDER} exists. Changing permissions"
   chown -R ${USERNAME}:${USERNAME} ${ZED_FOLDER}
   echo " * ZED SDK available:"
   ${ZED_FOLDER}tools/ZED_Explorer -v
else
   echo " * $ZED_FOLDER does not exist. ZED SDK not installed"
   exit 1
fi

# Build zed_wrapper if available
if [ -d "${ISAAC_ROS_WS}/src/zed-ros2-wrapper" ]; then
   echo ""
   echo "Installing dependencies for zed_wrapper..."
   rosdep install --from-paths ${ISAAC_ROS_WS}/src/zed-ros2-wrapper \
      --ignore-src -r -y || true
   echo "✓ rosdep dependencies for zed_wrapper installed"
   
   echo "Building zed_wrapper..."
   source /opt/ros/humble/setup.bash
   cd ${ISAAC_ROS_WS}/ && source ${ISAAC_ROS_WS}/install/setup.bash 2>/dev/null || true
   # sudo chown -R ${USERNAME}:${USERNAME} ${ISAAC_ROS_WS}/build ${ISAAC_ROS_WS}/install ${ISAAC_ROS_WS}/log 2>/dev/null || true
   colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to zed_wrapper
   # Setup the environment variables
   echo source $(pwd)/install/local_setup.bash >> ~/.bashrc
   source ~/.bashrc

   echo "✓ zed_wrapper built successfully"
   echo ""
fi
