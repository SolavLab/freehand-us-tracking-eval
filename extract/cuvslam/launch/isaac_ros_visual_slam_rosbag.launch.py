# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# ---------------------------------------------------------------------------
# MODIFIED for the freehand-US probe-tracking comparison.
#
# Apache-2.0 section 4(b) requires modified files to carry prominent notice of
# the change, so: this file began as an NVIDIA Isaac ROS launch template and was
# substantially rewritten here. NVIDIA's copyright and licence above are
# retained and continue to apply; the modifications are
# Copyright (c) 2026 Zohar Oddes and Dana Solav, MIT licensed (see LICENSE).
#
# What was changed:
#   * Plays back a converted RealSense recording as a ROS 2 bag
#     (`ros2 bag play --rate 0.4 --clock`), with `use_sim_time: True`, so the
#     original acquisition timestamps are preserved.
#   * `--read-ahead-queue-size 3000` and raised cuVSLAM buffer sizes, to stop
#     the SLAM node dropping frames under load. With the 0.4 rate this is the
#     frame-loss mitigation the paper reports as a limitation for any real-time
#     claim.
#   * Requests the optimised trajectory when playback exits, via an
#     OnProcessExit handler.
#   * Publishes the static transforms for THIS RIG's measured RealSense stereo
#     baseline and IMU offsets. They are not defaults; change them for other
#     hardware.
# ---------------------------------------------------------------------------

import os

# The container workspace root, replacing hardcoded absolute paths so the launch
# file works in a container mounted elsewhere.
ISAAC_ROS_WS = os.environ.get("ISAAC_ROS_WS", "/workspaces/isaac_ros-dev")
import launch
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction, ExecuteProcess, RegisterEventHandler, OpaqueFunction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import subprocess
import time


def call_get_all_poses_service(context, *args, **kwargs):
    """
    Called when rosbag playback finishes.
    Calls the get_all_poses service and saves response to file.
    """
    print("[launch] Rosbag playback finished, retrieving all poses...")
    
    try:
        # Wait a moment for SLAM to finalize poses
        time.sleep(1)
        
        # Call the service
        result = subprocess.run(
            [
                'ros2', 'service', 'call',
                '/visual_slam/get_all_poses',
                'isaac_ros_visual_slam_interfaces/srv/GetAllPoses', '{max_count: 20000}'
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("[launch] Service call succeeded, saving response...")
            
            # Save the raw response for parsing by the bash script
            # The response is in Python repr format from ros2 service call
            response_content = result.stdout.strip()
            
            if response_content:
                # Try multiple locations for writing
                output_files = [
                    '/tmp/poses_response.txt',
                    # '/home/admin/poses_response.txt',
                    # os.path.join(ISAAC_ROS_WS, 'poses_response.txt')
                ]
                
                written = False
                for output_file in output_files:
                    try:
                        with open(output_file, 'w') as f:
                            f.write(response_content)
                        print(f"[launch] Poses saved to {output_file}")
                        written = True
                        break
                    except Exception as e:
                        print(f"[launch] Could not write to {output_file}: {e}")
                
                if not written:
                    print("[launch] WARNING: Could not write poses to any location")
            else:
                print("[launch] WARNING: No response content from service call")
        else:
            print(f"[launch] Service call failed with return code {result.returncode}")
            if result.stderr:
                print(f"[launch] Error: {result.stderr}")
            if result.stdout:
                print(f"[launch] Output: {result.stdout[:200]}")
    except subprocess.TimeoutExpired:
        print("[launch] Service call timed out (>30s)")
    except Exception as e:
        print(f"[launch] Exception calling service: {e}")


def generate_launch_description():
    rosbag_path = LaunchConfiguration('rosbag_path')
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_imu_fusion = LaunchConfiguration('enable_imu_fusion')
    use_realsense = LaunchConfiguration('use_realsense')

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('isaac_ros_visual_slam'),
                'launch',
                'isaac_ros_visual_slam_realsense.launch.py'
            ])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'rectified_images': 'True',
            'enable_imu_fusion': enable_imu_fusion
        }.items(),
        condition=IfCondition(use_realsense)
    )
    
    # camera_link → camera_infra1_frame (identity)
    static_tf_infra1_frame = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0',
            '--qx', '0',
            '--qy', '0',
            '--qz', '0',
            '--qw', '1',
            '--frame-id', 'camera_link',
            '--child-frame-id', 'camera_infra1_frame',
        ],
        output='screen',
        condition=UnlessCondition(use_realsense)
    )
    # camera_aligned_depth_to_infra1_frame → camera_infra1_optical_frame (rotation only)
    # static_tf_aligned_infra1 = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     arguments=[
    #         '--x', '0',
    #         '--y', '0',
    #         '--z', '0',
    #         '--qx', '-0.5',
    #         '--qy', '0.5',
    #         '--qz', '-0.5',
    #         '--qw', '0.5',
    #         '--frame-id', 'camera_aligned_depth_to_infra1_frame',
    #         '--child-frame-id', 'camera_infra1_optical_frame',
    #     ],
    #     output='screen',
    #     condition=UnlessCondition(use_realsense)
    # )
    # camera_link → camera_aligned_depth_to_infra1_frame (identity)
    # static_tf_depth_to_infra1 = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     arguments=[
    #         '--x', '0',
    #         '--y', '0',
    #         '--z', '0',
    #         '--qx', '0',
    #         '--qy', '0',
    #         '--qz', '0',
    #         '--qw', '1',
    #         '--frame-id', 'camera_link',
    #         '--child-frame-id', 'camera_aligned_depth_to_infra1_frame',
    #     ],
    #     output='screen',
    #     condition=UnlessCondition(use_realsense)
    # )

    ## TF from camera_infraX_frame to camera_infraX_optical_frame (rotations only)
    static_tf_left_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0',
            '--qx', '-0.5',
            '--qy', '0.5',
            '--qz', '-0.5',
            '--qw', '0.5',
            '--frame-id', 'camera_infra1_frame',
            '--child-frame-id', 'camera_infra1_optical_frame',
        ],
        output='screen',
        condition=UnlessCondition(use_realsense)
    )
    static_tf_right_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0',
            '--qx', '-0.5',
            '--qy', '0.5',
            '--qz', '-0.5',
            '--qw', '0.5',
            '--frame-id', 'camera_infra2_frame',
            '--child-frame-id', 'camera_infra2_optical_frame',
        ],
        output='screen',
        condition=UnlessCondition(use_realsense)
    )
    # camera_link → camera_infra2_frame (with Y-axis baseline)
    static_tf_right_frame = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0',
            '--y', '-0.09495820850133896',
            '--z', '0',
            '--qx', '0',
            '--qy', '0',
            '--qz', '0',
            '--qw', '1',
            '--frame-id', 'camera_link',
            '--child-frame-id', 'camera_infra2_frame',
        ],
        output='screen',
        condition=UnlessCondition(use_realsense)
    )
    static_tf_accel_frame = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '-0.016019999980926514',
            '--y', '-0.030220000073313713',
            '--z', '0.007400000002235174',
            '--qx', '0',
            '--qy', '0',
            '--qz', '0',
            '--qw', '1',
            '--frame-id', 'camera_link',
            '--child-frame-id', 'camera_accel_frame',
        ],
        output='screen',
        condition=UnlessCondition(use_realsense)
    )
    static_tf_accel_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0',
            '--qx', '-0.5',
            '--qy', '0.5',
            '--qz', '-0.5',
            '--qw', '0.5',
            '--frame-id', 'camera_accel_frame',
            '--child-frame-id', 'camera_accel_optical_frame',
        ],
        output='screen',
        condition=UnlessCondition(use_realsense)
    )
    static_tf_gyro_frame = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '-0.016019999980926514',
            '--y', '-0.030220000073313713',
            '--z', '0.007400000002235174',
            '--qx', '0',
            '--qy', '0',
            '--qz', '0',
            '--qw', '1',
            '--frame-id', 'camera_link',
            '--child-frame-id', 'camera_gyro_frame',
        ],
        output='screen',
        condition=UnlessCondition(use_realsense)
    )
    static_tf_gyro_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0',
            '--qx', '-0.5',
            '--qy', '0.5',
            '--qz', '-0.5',
            '--qw', '0.5',
            '--frame-id', 'camera_gyro_frame',
            '--child-frame-id', 'camera_gyro_optical_frame',
        ],
        output='screen',
        condition=UnlessCondition(use_realsense)
    )

    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'enable_image_denoising': True,
            'rectified_images': True,
            'enable_imu_fusion': enable_imu_fusion,
            'use_sim_time': use_sim_time,
            # 'gyro_noise_density': 0.000244,
            # 'gyro_random_walk': 0.000019393,
            # 'accel_noise_density': 0.001862,
            # 'accel_random_walk': 0.003,
            # 'calibration_frequency': 200.0,
            'image_jitter_threshold_ms': 34.0,
            'base_frame': 'camera_link',
            'imu_frame': 'camera_gyro_optical_frame',
            'map_frame': 'map',
            'enable_slam_visualization': False,
            'enable_landmarks_view': False,
            'enable_observations_view': False,
            'enable_localization_n_mapping': True,
            'num_cameras': 2,
            'enable_debug_mode': False,
            'debug_dump_path': os.path.join(ISAAC_ROS_WS, 'visual_slam_debug') + '/',
            'image_buffer_size': 100,
            'imu_buffer_size': 500,
            'image_qos': 'DEFAULT',
            'imu_qos': 'DEFAULT',
            'camera_optical_frames': [
                'camera_infra1_optical_frame',
                'camera_infra2_optical_frame',
            ],
            'path_max_size': 1, # The maximum size of the buffer for pose trail visualization.
            'slam_max_map_size': 2000,
            # 'slam_throttling_time_ms': 100
        }],
        remappings=[
            ('visual_slam/image_0', 'camera/left/image_raw'),
            ('visual_slam/camera_info_0', 'camera/left/camera_info'),
            ('visual_slam/image_1', 'camera/right/image_raw'),
            ('visual_slam/camera_info_1', 'camera/right/camera_info'),
            ('visual_slam/imu', 'imu'),
        ],
    )

    visual_slam_container = ComposableNodeContainer(
        name='visual_slam_launch_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[visual_slam_node],
        output='screen',
        condition=UnlessCondition(use_realsense)
    )

    rosbag_execute = ExecuteProcess(
         cmd=[
            'ros2', 'bag', 'play', rosbag_path,
            '--topics', '/camera/left/image_raw', '/camera/right/image_raw', '/camera/left/camera_info', '/camera/right/camera_info', '/imu',
            '--read-ahead-queue-size', '3000',
            '--rate', '0.4',
            '--clock',
            # '--qos-profile-overrides-path', os.path.join(ISAAC_ROS_WS, 'qos_profiles.yaml')
        ],
        output='screen'
    )

    rosbag_player = TimerAction(
        period=4.0,
        actions=[rosbag_execute]
    )

    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'rosbag_path',
            description='Path to the rosbag directory (REQUIRED)'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulated time from rosbag'),
        DeclareLaunchArgument(
            'enable_imu_fusion',
            default_value='True',
            description='Forwarded to the Realsense launch to toggle IMU fusion'),
        DeclareLaunchArgument(
            'use_realsense',
            default_value='False',
            description='Launch RealSense driver alongside rosbag playback'),
        realsense_launch,
        static_tf_infra1_frame,
        # static_tf_aligned_infra1,
        # static_tf_depth_to_infra1,
        static_tf_right_frame,
        static_tf_left_optical,
        static_tf_right_optical,
        static_tf_accel_frame,
        static_tf_accel_optical,
        static_tf_gyro_frame,
        static_tf_gyro_optical,
        visual_slam_container,
        rosbag_player,
        RegisterEventHandler(
            OnProcessExit(
                target_action=rosbag_execute,
                on_exit=[
                    OpaqueFunction(function=call_get_all_poses_service),
                    launch.actions.EmitEvent(event=launch.events.Shutdown())
                ]
            )
        )
    ])
